import json, httplib2, couchdb, time, random, urllib

CDB = "deezer-mood-tracks"
FILE = '../data/deezer_tracks.json'
URI = 'https://api.deezer.com/track/'
SND_DIR = '/Users/alo/snd/deezer-moodplay/'

class FileDownloader:
    def __init__(self):
        server = couchdb.Server()
        self.cdb = server[CDB]

    def load_json(self):
        with open(FILE) as load_file:
            self.tracks = json.load(load_file)
            load_file.close()

    def get_uris(self):
        self.load_json()
        for filename in self.tracks:
            track = self.tracks[filename]
            doc = self.cdb.get(track["mbid"])
            if doc is None:
                uri = self.get_track(track["deezer_id"])
                if uri:
                    track["preview"] = uri
                track["_id"] = track["mbid"]
                del track["mbid"]
                self.cdb.save(track)
                if "preview" in track:
                    print track["_id"], track["preview"]
                else:
                    print track["_id"]
                time.sleep(round(random.uniform(0.5, 2.0), 3))

    def get_soundfiles(self):
        for row in self.cdb.view("views/tracks_with_preview"):
            track = row.value
            path = SND_DIR + track["_id"] + ".mp3"
            urllib.urlretrieve(track["preview"], path)
            print track["_id"], track["preview"]
            time.sleep(round(random.uniform(1.0, 3.0), 3))

    def get_track(self, deezer_id):
        uri = URI + str(deezer_id)
        re, co = httplib2.Http().request(uri)
        if re.status == 200:
            track = json.loads(co)
            if "id" in track and int(track["id"]) == deezer_id and "preview" in track:
                return track["preview"]
            else:
                return ""

class DeezerDbWriter:
    def __init__(self):
        srv = couchdb.Server()
        self.cdb = srv[CDB]

    def write(self):
        self.tracks = {}
        for res in self.cdb.view("views/tracks_with_preview"):
            track = res.value
            self.tracks[track["filename"]] = track
        with open('../data/tracks_with_preview.json', 'w') as write_file:
            write_file.write(json.dumps(self.tracks))
            write_file.close()

# FileDownloader().get_uris()
# DeezerDbWriter().write()
FileDownloader().get_soundfiles()
