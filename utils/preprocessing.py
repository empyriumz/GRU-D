import pandas as pd
import pickle  
class MortalityDataLoader:
    def __init__(self, dataset_dir, listfile=None, partial_data=0, debug=0):
        self._dataset_dir = dataset_dir
        if debug:
            with open("./data/debug-sample.pkl", "rb") as sample:
                data = pickle.load(sample)
            self._data = data["data"]
            self.pos_weight = data["pos_weight"]
        else:
            df = pd.read_csv(listfile)
            data = {"x": [], "last_x":[], "y": [], "interval": [], "mask": []}
            names = df['stay'].values
            data["y"] = df['y_true'].values
            for file_name in names:
                tmp_df = pd.read_csv(self._dataset_dir+"/"+file_name)
                tmp_df = tmp_df.dropna(how="all")
                tmp_data = tmp_df[["Hours", "Diastolic blood pressure"]].rename(columns={"Diastolic blood pressure": "x"})
                tmp_data = tmp_data.sort_values(by=["Hours"])
                tmp_data["interval"] = tmp_data["Hours"].diff().fillna(0)
                tmp_data["mask"] = tmp_data["x"].apply(lambda x: 1 if x == x else 0)
                tmp_data["x"] = tmp_data["x"].fillna(0)
                tmp_data["last_x"] = tmp_data["x"].shift().fillna(0)         
                data["x"].append(tmp_data["x"].values)
                data["last_x"].append(tmp_data["last_x"].values)
                data["interval"].append(tmp_data["interval"].values)
                data["mask"].append(tmp_data["mask"].values)
                if partial_data and len(data["x"]) == 1024:
                    data["y"] = data["y"][:1024]
                    break
            
            self._data = data
            # pos_weight is the ratio of (# of neg / # of pos) samples
            self.pos_weight = (len(data["y"]) - sum(data["y"])) / sum(data["y"])
 

# sample_data_loader = MortalityDataLoader(
#         dataset_dir="/host/StageNet/mortality_data/train",
#         listfile="/host/StageNet/mortality_data/val-mortality.csv",
#         partial_data=1,
#     )
# sample_data = {}
# sample_data["data"] = sample_data_loader._data 
# sample_data["pos_weight"] = sample_data_loader.pos_weight
# with open("../data/debug-sample.pkl", "wb") as handle:
#     pickle.dump(sample_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
