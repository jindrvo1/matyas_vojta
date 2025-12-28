import numpy as np
import pandas as pd
from rapidfuzz import process

from lib.registrations import Registrations


class Postprocessor:
    valid_registrations: pd.DataFrame

    OCR_NUMBER_TO_LETTER = {
        "O": ["0", "Θ"],
        "I": ["1", "|"],
        "L": ["1", "|"],
        "Z": ["2"],
        "E": ["3"],
        "A": ["4"],
        "S": ["5"],
        "G": ["6"],
        "T": ["7"],
        "B": ["8"],
        "P": ["9"],
    }

    OCR_LETTER_TO_NUMBER = {
        "0": ["O", "Q", "D", "Θ"],
        "1": ["I", "L", "|"],
        "2": ["Z"],
        "3": ["E"],
        "4": ["A"],
        "5": ["S"],
        "6": ["G"],
        "7": ["T"],
        "8": ["B"],
        "9": ["P", "g"],
    }

    OCR_REGISTRATION_PREFIX_MAPPING = {
        "K": ["X", "R", "H", "Y", "N"],
        "O": ["0", "Q", "D", "U", "C", "G", "Θ"],
        "M": ["N", "W", "H"],
    }

    VALID_VEHICLE_CATEGORIES = {
        "planes": ["AVREG_DATA.CATEGORIES.AIRPLANE"],
        "gliders": [
            "AVREG_DATA.CATEGORIES.GLIDER",
            "AVREG_DATA.CATEGORIES.POWERED_GLIDER",
        ],
    }

    VALID_COUNTRY_CODES = ["OK", "MK"]

    def __init__(self, path_to_registrations: str = "registrations.csv"):
        registrations = Registrations(path_to_registrations)
        if registrations.df is None:
            registrations.fetch_registrations_cz(1000)

        self.valid_registrations = (
            registrations.df if registrations.df is not None else pd.DataFrame()
        )

    def process_registration(self, registration: str) -> str:
        country_code, reg_number = self.correct_country_code_and_reg_number(
            registration
        )
        return f"{country_code}-{reg_number}"

    def correct_country_code_and_reg_number(self, registration: str) -> tuple[str, str]:
        if pd.isna(registration):
            return "", ""

        sep = "-"
        if sep in registration:
            reg_split = registration.split(sep)
        elif " " in registration:
            reg_split = registration.split(" ")
        else:
            reg_split = [registration[:2], registration[2:]]

        country_code, *reg_number_list = [reg for reg in reg_split if len(reg) > 0]
        reg_number = sep.join(reg_number_list)

        country_code_fixed = self._fix_country_code(country_code)
        valid_registrations = self.valid_registrations[
            self.valid_registrations["country_prefix"] == country_code_fixed
        ]
        valid_registrations = (
            valid_registrations
            if not valid_registrations.empty
            else self.valid_registrations
        )

        reg_number_fixed = self._match_reg_number(reg_number, valid_registrations)

        return country_code_fixed, reg_number_fixed

    def _match_reg_number(
        self, registration: str, valid_registrations: pd.DataFrame
    ) -> str:
        if pd.isna(registration):
            return ""

        digit_like, letter_like = self._classify_token(registration)

        if len(registration) <= 2 or len(registration) >= 5:
            return registration
        # assume airplane, or an airplane with an extra character, hence 3x A-Z
        elif (len(registration) == 3) or (
            len(registration) == 4 and letter_like > 0.7 and digit_like < 0.7
        ):
            registration = self._map_chars(registration, self.OCR_NUMBER_TO_LETTER)
            valid_registrations = valid_registrations[
                valid_registrations["category"].isin(
                    self.VALID_VEHICLE_CATEGORIES["planes"]
                )
            ]
        # either gliders, hence 4x 0-9
        elif len(registration) == 4:
            registration = self._map_chars(registration, self.OCR_LETTER_TO_NUMBER)
            valid_registrations = valid_registrations[
                valid_registrations["category"].isin(
                    self.VALID_VEHICLE_CATEGORIES["gliders"]
                )
            ]

        reg_match, _ = self._fuzzy_match(
            registration, valid_registrations["registration_number"].tolist()
        )

        return reg_match

    def _map_char(self, letter: str, mapping: dict[str, list[str]]) -> str:
        if len(letter) != 1:
            return letter

        for k, v in mapping.items():
            if letter in v:
                return k

        return letter

    def _map_chars(self, string: str, mapping: dict[str, list[str]]) -> str:
        return "".join([self._map_char(letter, mapping) for letter in string])

    def _fix_country_code(self, country_code: str) -> str:
        if country_code in self.VALID_COUNTRY_CODES:
            return country_code

        if len(country_code) <= 1:
            return country_code
        elif len(country_code) == 2:
            country_code_fixed = self._map_chars(
                country_code, self.OCR_REGISTRATION_PREFIX_MAPPING
            )
            return (
                country_code_fixed
                if country_code_fixed in self.VALID_COUNTRY_CODES
                else country_code
            )
        elif len(country_code) == 3:
            options = [country_code[0:2], country_code[1:3]][::-1]
            for option in options:
                option_fixed = self._fix_country_code(option)
                if option_fixed in self.VALID_COUNTRY_CODES:
                    return option
            return country_code

        return country_code

    def _map_letter_to_number(self, letter: str) -> str:
        return self._map_char(letter, self.OCR_LETTER_TO_NUMBER)

    def _map_number_to_letter(self, number: str) -> str:
        return self._map_char(number, self.OCR_NUMBER_TO_LETTER)

    def _char_type_score(
        self, char: str, similar_score: float = 0.5
    ) -> tuple[float, float]:
        char = char.upper()

        digit_like = [val for lst in self.OCR_NUMBER_TO_LETTER.values() for val in lst]
        letter_like = [val for lst in self.OCR_LETTER_TO_NUMBER.values() for val in lst]

        digit_score = (
            1.0 if char.isdigit() else similar_score if char in digit_like else 0.0
        )
        letter_score = (
            1.0 if char.isalpha() else similar_score if char in letter_like else 0.0
        )

        return digit_score, letter_score

    def _classify_token(self, token: str) -> tuple[float, float]:
        token = token.replace("-", "").replace(" ", "")

        digit_total = 0.0
        letter_total = 0.0

        for ch in token:
            dig, let = self._char_type_score(ch)
            digit_total += dig
            letter_total += let

        length = max(len(token), 1)

        digit_ratio = digit_total / length
        letter_ratio = letter_total / length

        return digit_ratio, letter_ratio

    def _fuzzy_match(
        self, registration: str, valid_registrations: list[str]
    ) -> tuple[str, float]:
        res = process.extractOne(registration, valid_registrations)
        return (res[0], res[1]) if res is not None else ("", np.nan)
