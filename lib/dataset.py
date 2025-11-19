import os
from typing import Optional

import pandas as pd

from lib.logger import Loggable


class Dataset(Loggable):
    df: pd.DataFrame

    def __init__(
        self,
        path_to_csv: str,
        data_folder: str = "data",
        drop_duplicates: bool = True,
        drop_missing_registrations: bool = True,
    ):
        super().__init__()
        self.df = pd.read_csv(path_to_csv)
        self.df = self._drop_duplicates(self.df) if drop_duplicates else self.df
        self.df = (
            self._drop_missing_registrations(self.df)
            if drop_missing_registrations
            else self.df
        )
        self.df = (
            self._fix_video_file_paths(self.df, data_folder) if data_folder else self.df
        )
        self.df = self._convert_segment_times(self.df)
        self.df = self._construct_index(self.df)

    def add_results(
        self,
        row_idx: tuple[int, int],
        ocr_result: dict[str, float],
        registration_col: str = "Predicted registration",
        confidence_col: str = "Prediction confidence",
    ):
        ((res_text, res_conf),) = ocr_result.items()
        self.df.loc[row_idx, registration_col] = res_text
        self.df.loc[row_idx, confidence_col] = res_conf

    def _drop_duplicates(
        self, df: pd.DataFrame, subset: list[str] = ["Video file"]
    ) -> pd.DataFrame:
        return df.drop_duplicates(subset=["Video file"], keep="first")

    def _drop_missing_registrations(
        self, df: pd.DataFrame, column: str = "Registration"
    ) -> pd.DataFrame:
        return df.dropna(subset=[column])

    def _fix_video_file_paths(self, df: pd.DataFrame, data_folder: str) -> pd.DataFrame:
        df["Video file"] = df["Video file"].apply(
            lambda file_name: os.path.join(data_folder, file_name)
        )
        return df

    def _convert_segment_times(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Segment start"] = (
            df["Segment start"]
            .apply(self._time_to_seconds)
            .apply(lambda t: max(t - 1, 0))
        )
        df["Segment end"] = df["Segment end"].apply(self._time_to_seconds) + 1
        return df

    def _time_to_seconds(self, timestr: Optional[str]) -> Optional[int]:
        if pd.isna(timestr):
            return timestr

        parts = [int(part) for part in timestr.split(":")]
        return sum([part * (60**i) for i, part in enumerate(parts[::-1])])

    def _construct_index(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = df["Video file"].str.extract(r".*_(\d+)\..*", expand=False).astype(int)
        df["ID"] = ids
        df["Segment"] = df.groupby("ID").cumcount()
        df = df.set_index(["ID", "Segment"])
        return df

    def __getattr__(self, name: str):
        return getattr(self.df, name)

    def __getitem__(self, key):
        return self.df[key]

    def __setitem__(self, key, value):
        self.df[key] = value

    def __len__(self):
        return len(self.df)

    @property
    def loc(self):
        class _LocWrapper:
            def __init__(self, df: pd.DataFrame):
                self.df = df

            def __getitem__(self, key: int | tuple[int, int]):
                result = self.df.loc[key]

                if not isinstance(result, pd.DataFrame):
                    return result

                return result.iloc[0] if result.shape[0] == 1 else result

        return _LocWrapper(self.df)
