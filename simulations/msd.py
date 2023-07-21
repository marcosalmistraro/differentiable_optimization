# -*- coding: utf-8 -*-
#
#  msd.py: Mass spectra deconvolution
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import copy
import json
from typing import List, Any, Dict

import cvxpy 
from LP.LP_linprog import LP_linprog
from QP.QP_qpth import qpth
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor


class Spectrum:

    def __init__(self, formula: str, ratios: np.ndarray, intensities: np.ndarray):
        self.formula: str = formula
        self.ratios: np.ndarray = ratios
        self.intensities: np.ndarray = intensities
        self.metadata: Dict[str, Any] = {}

    def copy(self) -> 'Spectrum':
        return copy.deepcopy(self)

    def __iadd__(self, other: 'Spectrum') -> 'Spectrum':
        self.ratios = np.concatenate((self.ratios, other.ratios), axis=0)
        self.intensities = np.concatenate((self.intensities, other.intensities), axis=0)
        idx = np.argsort(self.ratios)
        self.ratios = self.ratios[idx]
        self.intensities = self.intensities[idx]
        return self

    def __add__(self, other: 'Spectrum') -> 'Spectrum':
        formula = f'{self.formula} + {other.formula}'
        new_spectrum = Spectrum(formula, np.copy(self.ratios), np.copy(self.intensities))
        new_spectrum.__iadd__(other)
        return new_spectrum

    def __len__(self) -> int:
        return len(self.intensities)


class MSDProblemInstance:

    def __init__(
            self,
            theoretical_spectra: List[Spectrum],
            empirical_spectrum: Spectrum,
            p: np.ndarray
    ):
        spectra = theoretical_spectra + [empirical_spectrum]
        xs = np.concatenate([s.ratios for s in spectra], axis=0)
        ys = np.zeros((len(spectra), len(xs)))
        offset = 0
        for i, spectrum in enumerate(spectra):
            ys[i, offset:offset+len(spectrum)] = spectrum.intensities
            offset += len(spectrum.intensities)

        idx = np.argsort(xs)
        xs = xs[idx]
        ys = ys[:, idx]

        self.s: np.ndarray = xs
        self.mu: np.ndarray = ys[:-1, :]
        self.nu: np.ndarray = ys[-1, :]
        self.p: np.ndarray = p

    @staticmethod
    def solve_with_cvxpy(f: np.ndarray, g: np.ndarray, s: np.ndarray) -> np.ndarray:

        n = len(s)
        k = len(f)

        d = s[1:] - s[:-1]

        p = cvxpy.Variable(k, pos=True)
        t = cvxpy.Variable(n)

        constraints = [
            cvxpy.sum(p) == 1,
            -t + f.T @ p <= g,
            -t - f.T @ p <= -g
        ]
        objective = cvxpy.Minimize(t[:-1] @ d)
        prob = cvxpy.Problem(objective, constraints)
        _ = prob.solve(solver='ECOS')

        return p.value
    
    @staticmethod
    def solve_with_cvxpy_diff(q : Tensor,
                              A_in : Tensor,
                              b_in : Tensor, 
                              A_eq : Tensor, 
                              b_eq : Tensor) -> tuple:
        
        x = cvxpy.Variable(A_in.size()[1], pos=True)

        constraints = [ 
            A_in @ x <= b_in,
            A_eq @ x == b_eq
        ]

        objective = cvxpy.Minimize(q @ x)
        prob = cvxpy.Problem(objective, constraints)
        _ = prob.solve(solver='ECOS')
        lamda = constraints[0].dual_value
        nu = constraints[1].dual_value

        return x.value, lamda, nu

    def show(self):
        plt.subplot(2, 1, 1)
        for i in range(len(self.mu)):
            plt.bar(self.s, self.mu[i, :])
            plt.title('Theoretical spectra')
        plt.subplot(2, 1, 2)
        plt.bar(self.s, self.nu)
        plt.title('Empirical spectrum')
        plt.show()


class MoNA:

    def __init__(self, filepath: str):

        self.spectra: Dict[str, List[Spectrum]] = {}

        with open(filepath, 'r') as f:
            data = json.load(f)

        for entry in data:

            spectrum = np.asarray([element.split(':') for element in entry['spectrum'].split()], dtype=float)
            spectrum = Spectrum('unknown', spectrum[:, 0], spectrum[:, 1])

            # Get metadata
            inchi = 'unknown'
            for metadata in entry['metaData']:
                if metadata['name'] == 'molecular formula':
                    spectrum.formula = metadata['value']
                if metadata['name'] == 'exact mass':
                    spectrum.metadata['mass'] = float(metadata['value'])
                if metadata['name'] == 'ionization':
                    spectrum.metadata['ionization'] = metadata['value']
                if metadata['name'] == 'ionization mode':
                    spectrum.metadata['ionization-mode'] = metadata['value']
                if metadata['name'] == 'instrument':
                    spectrum.metadata['instrument'] = metadata['value']
                if metadata['name'] == 'instrument type':
                    spectrum.metadata['instrument-type'] = metadata['value']
                if metadata['name'] == 'fragmentation mode':
                    spectrum.metadata['fragmentation-mode'] = metadata['value']
            for compound in entry['compound']:
                if compound['kind'] == 'biological':
                    for metadata in compound['metaData']:
                        if metadata['name'] == 'molecular formula':
                            spectrum.formula = metadata['value']
                    inchi = compound['inchi']

            # Get mass spectrum
            if inchi not in self.spectra:
                self.spectra[inchi] = []
            self.spectra[inchi].append(spectrum)

        spectra = []
        for key in self.spectra.keys():
            if len(self.spectra[key]) > 1:
                positive, negative = None, None
                for spectrum in self.spectra[key]:
                    if spectrum.metadata['ionization-mode'] == 'positive':
                        positive = spectrum
                    elif spectrum.metadata['ionization-mode'] == 'negative':
                        negative = spectrum
                if (positive is not None) and (negative is not None):
                    spectra.append((positive, negative))
        self.spectra = spectra

    def random(self, n_compounds: int = 5) -> MSDProblemInstance:
        idx = np.random.randint(0, len(self.spectra), size=n_compounds)
        alpha = np.random.rand(n_compounds)
        alpha /= np.sum(alpha)

        # Theoretical spectra
        theoretical_spectra = [self.spectra[i][1].copy() for i in idx]
        for spectrum in theoretical_spectra:
            spectrum.intensities /= np.sum(spectrum.intensities)

        # Empirical spectrum
        spectra = [self.spectra[i][0].copy() for i in idx]
        for i, spectrum in enumerate(spectra):
            weight = spectrum.metadata['mass'] / np.dot(spectrum.intensities, spectrum.ratios)
            spectrum.intensities *= alpha[i] * weight
        empirical_spectrum = spectra[0]
        for i in range(1, len(spectra)):
            empirical_spectrum += spectra[i]
        empirical_spectrum.intensities /= np.sum(empirical_spectrum.intensities)

        return MSDProblemInstance(theoretical_spectra, empirical_spectrum, alpha)
