import couchdb, os, json
# import madmom
import librosa
import numpy as np

FPS = 100

class FeatureExtractor:
    def __init__(self):
        self.audio_path = '/Users/alo/snd/deezer-mood-test/'
        # self.audio_path = '/Users/alo/snd/deezer-moodplay/'
        self.server = couchdb.Server()
        self.db = self.server['moodplay-features-merged']

    def run(self):
        for filename in os.listdir(self.audio_path):
            fullpath = os.path.join(self.audio_path, filename)
            # if os.path.getsize(fullpath) > 1000 and filename > "00ac8821-8e3a-4cc1-a160-e17e5a296611.mp3":
            if True:
                json["_id"] = filename.split(".")[0]
                # json = self.extract_madmom(fullpath)
                json = self.extract_loudness(fullpath)
                # self.db.save(json)
                print(json["_id"])
                # print(json)
                # print("\n\n\n")

    def extract_madmom(self, path):
        signal = madmom.audio.signal.Signal(path)
        key = self.extract_key(signal)
        chords = self.extract_chords(signal)
        tempo = self.extract_tempo(signal)
        beats = self.extract_beats(signal)
        # mfcc = self.extract_mfcc(path)
        return {
            "key": key,
            "chords": [ { "start": round(c[0], 4), "end": round(c[1], 4), "chord": c[2] } for c in chords.tolist()],
            "tempo": [ { "tempo": t[0], "strength": t[1] } for t in tempo.tolist() if t[1] > 0.1 ],
            "beats": [ { "start": b[0], "position": int(b[1]) } for b in beats.tolist() ],
            # "mfcc": mfcc.tolist()
        }

    def extract_key(self, signal):
        proc = madmom.features.key.CNNKeyRecognitionProcessor()
        estimates = proc(signal)
        return madmom.features.key.key_prediction_to_label(estimates)

    def extract_chords(self, signal):
        dcp = madmom.audio.chroma.DeepChromaProcessor()
        decode = madmom.features.chords.DeepChromaChordRecognitionProcessor()
        chords = madmom.processors.SequentialProcessor([dcp, decode])
        return chords(signal)

    def extract_beats(self, signal):
        proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        act = madmom.features.downbeats.RNNDownBeatProcessor()(signal)
        return proc(act)

    def extract_tempo(self, signal):
        proc = madmom.features.tempo.TempoEstimationProcessor(fps=FPS)
        act = madmom.features.beats.RNNBeatProcessor()(signal)
        return proc(act)

    def merge_db(self):
        mfdb = self.server["moodplay-features-mfcc"]
        merged = self.server["moodplay-features-merged"]
        for _id in self.db:
            doc = self.db.get(_id)
            mfcc = mfdb.get(_id)
            json = { "_id": _id  }
            json["beats"] = doc["beats"]
            json["chords"] = doc["chords"]
            json["key"] = doc["key"]
            json["tempo"] = doc["tempo"]
            json["mfcc"] = mfcc["librosa"]["mfcc"]
            merged.save(json)

class LibrosaExtractor:
    def __init__(self):
        self.audio_path = '/Users/alo/snd/deezer-moodplay/'
        self.server = couchdb.Server()
        self.db_from = self.server['moodplay-features-merged']
        self.db_to = self.server['moodplay-features']

    def run(self):
        for filename in os.listdir(self.audio_path):
            fullpath = os.path.join(self.audio_path, filename)
            if os.path.getsize(fullpath) > 1000: #and filename > "00ac8821-8e3a-4cc1-a160-e17e5a296611.mp3":
            # if True:
                _id = filename.split(".")[0]
                doc = self.db_from.get(_id)
                amp, mfcc = self.collect_features(fullpath, doc["beats"])
                json = { "_id": _id }
                json["beats"] = doc["beats"]
                json["chords"] = doc["chords"]
                json["key"] = doc["key"]
                json["tempo"] = doc["tempo"]
                json["amplitude"] = amp
                json["mfcc"] = mfcc
                self.db_to.save(json)
                print(_id)

    def collect_features(self, path, beats):
        audio, sr = librosa.load(path)
        amp = self.extract_loudness(audio)
        mfcc = self.extract_mfcc(audio, sr)
        n_frames = len(mfcc.T)
        b_amps = []
        b_mfcc = []
        total_dur = beats[0]['start'] + beats[-1]['start']
        for i in range(len(beats)-2):
            ths, nxt = beats[i:i+2]
            first = self.linlin(ths["start"], 0.0, total_dur, 0, n_frames)
            last = self.linlin(nxt["start"], 0.0, total_dur, 0, n_frames)
            fra = np.mean(amp[int(first):int(last)])
            frm = np.mean(mfcc.T[int(first):int(last)], axis=0)
            b_amps.append({
                "start": ths["start"],
                "value": float(np.around(fra, 5))
            })
            b_mfcc.append({
                "start": ths["start"],
                "value": frm.tolist()
            })
        return (b_amps, b_mfcc)

    def extract_mfcc(self, audio, sr):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return mfcc

    def extract_loudness(self, audio):
        S = librosa.stft(audio)
        power = np.abs(S)**2
        p_mean = np.sum(power, axis=0, keepdims=True)
        db = librosa.power_to_db(p_mean, ref=np.max(power))
        amp = librosa.db_to_amplitude(db)
        return amp[0]

    def linlin(self, v, imn, imx, omn, omx):
        return (v-imn)/(imx-imn)*(omx-omn)+omn;

class JsonWriter:
    def __init__(self):
        self.server = couchdb.Server()
        self.db = self.server['moodplay-features']
        self.path = "../data/features.json"

    def run(self):
        data = { }
        for _id in self.db:
            doc = self.db.get(_id)
            data[_id] = {
                "amplitude": doc["amplitude"],
                "beats": doc["beats"],
                "chords": doc["chords"],
                "key": doc["key"],
                "mfcc": doc["mfcc"],
                "tempo": doc["tempo"]
            }
        with open(self.path, "w") as wf:
            wf.write(json.dumps(data))
            wf.close()

# FeatureExtractor().run()
# FeatureExtractor().merge_db()
# LibrosaExtractor().run()
JsonWriter().run()
