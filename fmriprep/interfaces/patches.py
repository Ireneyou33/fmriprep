# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Temporary patches
-----------------

"""

from random import randint
from time import sleep

from numpy.linalg.linalg import LinAlgError
from niworkflows.nipype.algorithms import confounds as nac


class RobustACompCor(nac.ACompCor):
    """
    Runs aCompCor several times if it suddenly fails with
    https://github.com/poldracklab/fmriprep/issues/776

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super(RobustACompCor, self)._run_interface(runtime)
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))
            else:
                return runtime
