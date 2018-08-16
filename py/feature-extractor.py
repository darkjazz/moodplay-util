import couchdb, os
# import madmom, librosa

FPS = 100

class FeatureExtractor:
    def __init__(self):
        # self.audio_path = '/Volumes/FAST-VMs/snd/deezer-moodplay/'
        self.audio_path = '/Users/alo/snd/deezer-moodplay/'
        self.server = couchdb.Server()
        self.db = self.server['moodplay-features']

    def run(self):
        for filename in os.listdir(self.audio_path):
            fullpath = os.path.join(self.audio_path, filename)
            if os.path.getsize(fullpath) > 1000 and filename > "00ac8821-8e3a-4cc1-a160-e17e5a296611.mp3":
                json = self.extract_madmom(fullpath)
                json["_id"] = filename.split(".")[0]
                self.db.save(json)
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

    def extract_mfcc(self, path, hop):
        y, sr = librosa.load(path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return mfcc

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

# FeatureExtractor().run()
FeatureExtractor().merge_db()
