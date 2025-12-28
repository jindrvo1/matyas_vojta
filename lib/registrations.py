import pathlib
import re
import unicodedata
from functools import reduce
from typing import ClassVar, Self, TypedDict
from urllib.parse import urlencode

import pandas as pd
import requests as r
from camelot.core import Table
from camelot.io import read_pdf


class CountryConfig(TypedDict):
    base_url: str
    url_params: dict[str, str]
    certificate_path: str


class RegistrationAPIConfig:
    config: CountryConfig
    defaults: ClassVar[dict[str, CountryConfig]] = {
        "cz": {
            "base_url": "https://lr.caa.cz/api/avreg/filtered",
            "url_params": {
                "search": "",
                "order": "",
                "start": "",
                "length": "",
            },
            "certificate_path": "registrations/certs/cz_chain.pem",
        },
    }

    def __init__(self, config: CountryConfig):
        self.config = config

    def construct_url(self, url_params: dict[str, str | int] = {}) -> str:
        url_params = {k: str(v) for k, v in url_params.items()}
        url_params_merged = {**self.config["url_params"], **url_params}
        url_params_encoded = urlencode(url_params_merged)

        return f"{self.config['base_url']}?{url_params_encoded}"

    @classmethod
    def construct_cz(
        cls,
        config: CountryConfig | None = None,
    ) -> Self:
        return cls._construct(
            "cz",
            config=config,
        )

    @classmethod
    def _construct(
        cls,
        country: str,
        *,
        config: CountryConfig | None = None,
    ) -> Self:
        config_merged = cls.defaults[country]
        if config is not None:
            config_merged.update(config)

        return cls(config_merged)

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'base_url': {self.config.get('base_url')}, \n"
        res += f"\t'url_params': {self.config.get('url_params')}, \n"
        res += f"\t'certificate_path': {self.config.get('certificate_path')}\n"
        res += ")"

        return res


class Registrations:
    df: pd.DataFrame
    path_to_csv: str | None

    AlLOWED_COLS = [
        "aircraft_type",
        "country_prefix",
        "registration_number",
        "category",
    ]

    def __init__(self, path_to_csv: str | None = "registrations/registrations_cz.csv"):
        self.path_to_csv = path_to_csv

        if path_to_csv and pathlib.Path(path_to_csv).is_file():
            self.df = self._load_df(path_to_csv)
        else:
            self.df = pd.DataFrame(columns=self.AlLOWED_COLS)

    def _load_df(self, path_to_csv: str) -> pd.DataFrame:
        return pd.read_csv(path_to_csv)

    def to_csv(self, path_to_csv: str | None = None):
        path_to_csv = path_to_csv or self.path_to_csv
        self.df.to_csv(path_to_csv, index=False)

    def append_registrations_cz(self, page_length: int = 1000) -> Self:
        api_config = RegistrationAPIConfig.construct_cz()
        res: list[dict] = []
        retrieved_all = False

        while retrieved_all is False:
            url = api_config.construct_url({"start": len(res), "length": page_length})
            response = r.get(url, verify=api_config.config["certificate_path"])
            response_data = response.json()
            response_data = response_data["rows"]
            res += response_data
            retrieved_all = len(response_data) == 0

        df = pd.DataFrame(res)
        df = df[df["deletion_date"].isna()]
        df = df.rename(columns={"type": "aircraft_type"})
        df["country_prefix"] = "OK"
        df = df[self.AlLOWED_COLS]

        self.df = pd.concat([self.df, df], ignore_index=True)

        return self

    def append_registrations_sk(
        self,
        pdf_path: str,
        pages: str = "all",
        line_scale: int = 30,
    ) -> Self:
        tables = read_pdf(
            pdf_path, pages=pages, flavor="lattice", line_scale=line_scale
        )

        header_table, content_tables = tables[0], tables[1::2]
        header = self._parse_header_table(header_table)

        df = pd.DataFrame(columns=header)

        for table in content_tables:
            df_page = table.df
            df_page.columns = header
            df = pd.concat([df, df_page], ignore_index=True)

        df = self._process_df(df)

        self.df = pd.concat([self.df, df], ignore_index=True)

        return self

    def _replace_newlines(self, text: str | None, replace_with: str = "") -> str:
        return re.sub(r"\n", replace_with, text) if text else ""

    def _normalize_header_names(self, text: str | None) -> str:
        if text is None:
            return ""

        text = text.strip().lower()
        parts = text.split("/")
        text = parts[-1].strip()
        text = re.sub(r"\s+", "_", text)

        normalized = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in normalized if not unicodedata.combining(c))

        return text

    def _parse_header_table(self, table: Table) -> pd.Index:
        df = table.df
        df = df.map(self._replace_newlines)
        df = df.map(self._normalize_header_names)
        header = pd.Index(df.iloc[0])

        return header

    def _extract_reg_number(
        self, df_subset: pd.DataFrame, pattern: str = r".*OM\ *-\ *(.*)"
    ) -> pd.Series:
        extracted = [
            df_subset[col]
            .astype(str)
            .str.upper()
            .str.replace(r"\s+", " ")
            .str.extract(pattern, expand=False)
            for col in df_subset.columns
        ]

        res = reduce(lambda a, b: a.combine_first(b), extracted)

        return res

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.map(self._replace_newlines).map(str.strip)

        df["registration_number"] = self._extract_reg_number(
            df[["registration_marks", "aircraft_type"]]
        )
        df["country_prefix"] = "OM"
        df["category"] = None

        df = df[self.AlLOWED_COLS]

        return df

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'path_to_csv': {self.path_to_csv}, \n"
        res += f"\t'n_rows': {self.df.shape[0]}\n"
        res += ")"

        return res
