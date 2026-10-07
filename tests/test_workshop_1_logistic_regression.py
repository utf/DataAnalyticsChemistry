"""Check the logistic-regression lesson without completing the student notebook.

Run with: python -m unittest discover -s tests -v
Requires numpy, pandas, scikit-learn and matplotlib from requirements.txt.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOTEBOOK = Path(__file__).resolve().parents[1] / "workshop-1.ipynb"


class LogisticRegressionLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

    def setUp(self):
        self.printed = []
        self.namespace = {
            "np": np,
            "pd": pd,
            "plt": plt,
            "print": lambda *args: self.printed.append(args),
        }
        self.run_cell("from sklearn.datasets import load_iris")
        self.run_cell("from sklearn.linear_model import LogisticRegression")

    def run_cell(self, marker):
        matches = [source for source in self.code_cells if marker in source]
        self.assertEqual(len(matches), 1, f"Expected one cell containing {marker!r}")
        source = matches[0]
        # Supply the documented exercise answers only in memory. Leave all
        # executable lesson code, including the chosen prediction method, intact.
        for blank, answer in {
            "model.fit(  )": "model.fit(X, y)",
            "model.predict_proba(  )": "model.predict_proba(X)",
            "log_loss(  )": "log_loss(y, y_prob)",
        }.items():
            source = source.replace(blank, answer)
        exec(compile(source, f"{NOTEBOOK.name}: {marker}", "exec"), self.namespace)

    def assert_probability_loss(self):
        y = np.asarray(self.namespace["y"])
        probabilities = self.namespace["y_prob"]
        np.testing.assert_array_equal(self.namespace["model"].classes_, [0, 1])
        self.assertEqual(probabilities.shape, (len(y), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))
        # For y in {0, 1}, binary cross-entropy is the average negative log
        # probability assigned to the observed class, using natural logarithms.
        expected_loss = -np.log(probabilities[np.arange(len(y)), y]).mean()
        self.assertAlmostEqual(self.namespace["loss"], expected_loss, places=12)

    def test_one_feature_log_loss_uses_probabilities(self):
        self.run_cell("from sklearn.metrics import log_loss")
        self.assert_probability_loss()

    def test_two_feature_log_loss_uses_probabilities(self):
        self.run_cell("from sklearn.metrics import log_loss")
        # Discard the earlier probabilities so this check also catches a stale
        # one-feature prediction being reused by the two-feature lesson cell.
        del self.namespace["y_prob"]
        self.run_cell('X = df[["petal length (cm)", "petal width (cm)"]].values\nmodel = LogisticRegression()\nmodel.fit(  )')
        self.assert_probability_loss()
        np.testing.assert_allclose(
            self.namespace["y_prob"],
            self.namespace["model"].predict_proba(self.namespace["X"]),
        )

    def test_printed_coefficients_reconstruct_the_sigmoid(self):
        self.run_cell("print('β₀:', model.")
        parameters = dict(self.printed)
        beta0 = np.asarray(parameters["β₀:"]).item()
        beta1 = np.asarray(parameters["β₁:"]).item()
        x = self.namespace["X"][:, 0]
        reconstructed = 1 / (1 + np.exp(-beta0 - beta1 * x))
        probabilities = self.namespace["model"].predict_proba(self.namespace["X"])
        np.testing.assert_allclose(reconstructed, probabilities[:, 1], atol=1e-12)

    def test_decision_boundary_is_a_scalar_and_plot_renders(self):
        self.addCleanup(plt.close, "all")
        with patch.object(plt, "show"):
            self.run_cell("decision_boundary = X_new")
        boundary = self.namespace["decision_boundary"]
        self.assertTrue(np.isscalar(boundary))
        first_crossing = np.flatnonzero(self.namespace["y_prob"][:, 1] >= 0.5)[0]
        self.assertEqual(boundary, self.namespace["X_new"][first_crossing, 0])
        # Text-coordinate conversion is deferred until the figure is rendered.
        self.namespace["fig"].canvas.draw()


if __name__ == "__main__":
    unittest.main()
