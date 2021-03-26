import pandas as pd

class MortalityDataLoader:
    def __init__(self, dataset_dir, listfile=None, partial_data=0):
        self._dataset_dir = dataset_dir
        df = pd.read_csv(listfile)
        data = {"x": [], "y": [], "interval": [], "mask": []}
        names = df['stay'].values
        data["y"] = df['y_true'].values
        for file_name in names:
            tmp_df = pd.read_csv(self._dataset_dir+"/"+file_name)
            tmp_df = tmp_df.dropna(how="all")
            tmp_data = tmp_df[["Hours", "Diastolic blood pressure"]].rename(columns={"Diastolic blood pressure": "x"})
            tmp_data = tmp_data.sort_values(by=["Hours"])
            tmp_data["interval"] = tmp_data["Hours"].diff().fillna(0)
            tmp_data["mask"] = tmp_data.where(tmp_data["x"]>=0).astype(int)
            tmp_data["last_x"] = tmp_data["x"].fillna(method='ffill')
            data["x"].append(tmp_data["x"].values)
            data["last_x"].append(tmp_data["last_x"].values)
            data["interval"].append(tmp_data["interval"].values)
            data["mask"].append(tmp_data["mask"].values)
            if partial_data and len(data["x"]) == 1024:
                data["y"] = data["y"][:1024]
                break
        
        self._data = data