signal, sample_rate = librosa.load("Data/genres_original/metal/metal.00086.wav", sr=SAMPLE_RATE)
x = librosa.stft(signal, n_fft = 2048, hop_length = 512)
np.abs(x)

def save_signal_data_mag(dataset_path, json_path, n_fft=2048, hop_length=512, num_segments=10, n_mels = 128):
    data = {
        "mapping": [],
        "labels": [],
        "spec_mag": [],
        "index_val": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            # process all audio files in genre sub-dir
            for f in filenames:
                if f[-3:] != "wav":
                    continue
        		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    # extract spec
                    cur_signal = signal[start:finish]
                    #cur_spec = librosa.feature.melspectrogram(y = cur_signal, sr=SAMPLE_RATE, n_fft=n_fft, 
                    #                                 hop_length=hop_length, n_mels=n_mels)  
                    cur_spec = librosa.stft(cur_signal, n_fft = n_fft, hop_length = hop_length)
                    cur_spec_mag = librosa.amplitude_to_db(np.abs(cur_spec))
                    # store only mfcc feature with expected number of vectors
                    data["spec_mag"].append(cur_spec_mag.T.tolist())
                    data["labels"].append(i-1)
                    data["index_val"].append(f + "_" + str(d))
                print("-", end = "")
    # save to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def save_signal_data_phase(dataset_path, json_path, n_fft=2048, hop_length=512, num_segments=10):
    data = {
        "mapping": [],
        "labels": [],
        "spec_phase": [],
        "index_val": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            # process all audio files in genre sub-dir
            for f in filenames:
                if f[-3:] != "wav":
                    continue
        		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    # extract spec
                    cur_signal = signal[start:finish]
                    cur_spec = librosa.stft(cur_signal, n_fft = n_fft, hop_length = hop_length)
                    cur_spec_phase = np.angle(cur_spec)
                    # store only mfcc feature with expected number of vectors
                    data["spec_phase"].append(cur_spec_phase.T.tolist())
                    data["labels"].append(i-1)
                    data["index_val"].append(f + "_" + str(d))
                print("-", end = "")
    # save to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

print("Saving magnitudes")
save_signal_data_mag(DATASET_PATH, JSON_PATH_MAG)
print("Saving phases")
save_signal_data_phase(DATASET_PATH, JSON_PATH_MAG)


DATA_PATH = "path/to/dataset/in/json/file"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y