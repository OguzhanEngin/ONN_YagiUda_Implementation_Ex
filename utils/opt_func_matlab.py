import numpy as np
import pandas as pd

import matlab
import matlab.engine as matlab_engine


class AntennaFunction_v2:
    """Antenna objective value function:
    - Data about specific antenna
    - Connects to matlab for calculation of objective value
    - Lambda = 1.8
    """

    def __init__(self) -> None:
        self.num_directors = 4.0
        self.max_director_lengths = (
            np.array([0.495, 0.495, 0.495, 0.495]) * 1.8
        ).tolist()
        self.max_director_spacings = (np.array([0.45, 0.45, 0.45, 0.45]) * 1.8).tolist()

        self.max_reflector_length = 0.52 * 1.8
        self.max_dipole_length = 0.52 * 1.8
        self.max_reflector_spacing = 0.45 * 1.8

        self.min_director_lengths = (np.array([0.4, 0.4, 0.4, 0.4]) * 1.8).tolist()
        self.min_director_spacings = (np.array([0.15, 0.15, 0.15, 0.15]) * 1.8).tolist()

        self.min_reflector_length = 0.42 * 1.8
        self.min_dipole_length = 0.42 * 1.8
        self.min_reflector_spacing = 0.15 * 1.8

        self.director_length_cols = [
            "director_length_1",
            "director_length_2",
            "director_length_3",
            "director_length_4",
        ]
        self.director_spacing_cols = [
            "director_spacing_1",
            "director_spacing_2",
            "director_spacing_3",
            "director_spacing_4",
        ]
        self.reflector_length_col = "reflector_length"
        self.reflector_spacing_col = "reflector_spacing"
        self.dipole_length_col = "dipole_length"

        self.x_names = (
            self.director_length_cols
            + self.director_spacing_cols
            + [self.reflector_length_col]
            + [self.reflector_spacing_col]
            + [self.dipole_length_col]
        )
        self.x_max = (
            self.max_director_lengths
            + self.max_director_spacings
            + [self.max_reflector_length]
            + [self.max_reflector_spacing]
            + [self.max_dipole_length]
        )
        self.x_min = (
            self.min_director_lengths
            + self.min_director_spacings
            + [self.min_reflector_length]
            + [self.min_reflector_spacing]
            + [self.min_dipole_length]
        )
        self.n_dim = len(self.x_names)

        self.matlab_eng = matlab_engine.start_matlab()

    def calculate_val(self, input_ser: pd.Series):
        (
            director_length_arr,
            director_spacing_arr,
            reflector_length,
            reflector_spacing,
        ) = self._convert_features_matlab(input_ser)

        dipole_folded = self.matlab_eng.dipoleFolded(
            "Length",
            input_ser[self.dipole_length_col],
            "Width",
            0.0136 * 1.8,
            "Spacing",
            0.0061 * 1.8,
        )
        y = self.matlab_eng.yagiUda(
            "Exciter",
            dipole_folded,
            "NumDirectors",
            self.num_directors,
            "DirectorLength",
            matlab.double(director_length_arr),
            "DirectorSpacing",
            matlab.double(director_spacing_arr),
            "ReflectorLength",
            reflector_length,
            "ReflectorSpacing",
            reflector_spacing,
        )
        out = self.matlab_eng.pattern(y, 165e6)
        obj_val = np.array(out._data).ravel().max()

        return obj_val

    def calculate_batch(self, input_df: pd.DataFrame):
        """Calculate maximum yaiUda value in batch"""
        objective_values = input_df.apply(self.calculate_val, axis=1)
        return objective_values

    def _convert_features_matlab(self, input_ser: pd.Series):
        director_length_arr = input_ser[self.director_length_cols].tolist()
        director_spacing_arr = input_ser[self.director_spacing_cols].tolist()
        reflector_length = input_ser[self.reflector_length_col]
        reflector_spacing = input_ser[self.reflector_spacing_col]

        return (
            director_length_arr,
            director_spacing_arr,
            reflector_length,
            reflector_spacing,
        )
