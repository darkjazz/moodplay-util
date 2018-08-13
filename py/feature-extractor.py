import madmom, couchdb, os

FPS = 100

class FeatureExtractor:
    def __init__(self):
        self.audio_path = '/Volumes/FAST-VMs/snd/deezer-moodplay/'
        server = couchdb.Server()
        self.db = server['moodplay-features']

    def run(self):
        for filename in os.listdir(self.audio_path):
            fullpath = os.path.join(self.audio_path, filename)
            json = self.extract_madmom(fullpath)
            print(json)
            print("\n\n\n")

    def extract_madmom(self, path):
        signal = madmom.audio.signal.Signal(path)
        key = self.extract_key(signal)
        chords = self.extract_chords(signal)
        tempo = self.extract_tempo(signal)
        beats = self.extract_beats(signal)
        return {
            "key": key,
            "chords": [ { "start": round(c[0], 4), "end": round(c[1], 4), "chord": c[2] } for c in chords.tolist()],
            "tempo": [ { "tempo": t[0], "strength": t[1] } for t in tempo.tolist() if t[1] > 0.1 ],
            "beats": [ { "start": b[0], "position": int(b[1]) } for b in beats.tolist() ]
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

    def exract_mfcc(self, signal, beats):
        proc = madmom.features.downbeats.SyncronizeFeaturesProcessor(beats, FPS)
        mfcc = proc(signal)
        return mfcc

FeatureExtractor().run()
