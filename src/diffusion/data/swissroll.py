"""Swiss roll datamodule."""

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule


def make_swiss_roll_2d(
    num_samples: int,
    noise_level: float = 0.5,
    scaling: float = 0.15,
    random_state: int | None = None,
    test_size: float | int | None = None,
    num_bands: int | None = None,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Create 2D Swiss roll data.

    Parameters
    ----------
    num_samples : int
        Number of samples to create.
    noise_level : float
        Noise standard deviation.
    scaling : float
        Scaling parameter.
    random_state : int or None
        Random generator seed.
    test_size : float, int or None
        Test size parameter.
    num_bands : int or None
        If given, splits the spiral into this many concentric bands
        and assigns each sample a class label by alternating 0/1.

    """

    # create 3D data
    x, t = make_swiss_roll(num_samples, noise=abs(noise_level), random_state=random_state)

    # restrict to 2D
    x = x[:, [0, 2]]

    # scale data
    x = scaling * x

    # build band labels along the spiral parameter t
    if num_bands is not None:
        num_bands = abs(int(num_bands))
        if num_bands < 1:
            raise ValueError("num_bands must be at least 1")
        edges = np.linspace(t.min(), t.max(), num_bands + 1)
        band_idx = np.clip(np.digitize(t, edges) - 1, 0, num_bands - 1)
        y = (band_idx % 2).astype(np.int64)

    # return
    if test_size is None:
        if num_bands is None:
            return x
        return x, y

    # split data and return
    else:
        if num_bands is None:
            x_train, x_val = train_test_split(x, test_size=test_size)
            return x_train, x_val
        else:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)
            return x_train, x_val, y_train, y_val


class SwissRollDataModule(LightningDataModule):
    """
    DataModule for 2D Swiss roll data.

    Parameters
    ----------
    num_train : int
        Number of training samples.
    num_val : int
        Number of validation samples.
    num_test : int
        Number of testing samples.
    noise_level : float
        Noise standard deviation.
    scaling : float
        Scaling parameter.
    random_state : int
        Random generator seed.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    """

    def __init__(
        self,
        num_train: int,
        num_val: int = 0,
        num_test: int = 0,
        noise_level: float = 0.5,
        scaling: float = 0.15,
        random_state: int = 42,
        batch_size: int = 32,
        num_workers: int = 0,
        num_bands: int | None = None,
    ):
        super().__init__()

        # set data parameters
        self.num_train = abs(int(num_train))
        self.num_val = abs(int(num_val))
        self.num_test = abs(int(num_test))
        self.noise_level = abs(noise_level)
        self.scaling = scaling
        self.num_bands = None if num_bands is None else abs(int(num_bands))

        # set random state
        self.random_state = random_state

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Prepare numerical data."""

        # create data
        num_samples = self.num_train + self.num_val + self.num_test

        out = make_swiss_roll_2d(
            num_samples,
            noise_level=self.noise_level,
            scaling=self.scaling,
            random_state=self.random_state,
            test_size=None,
            num_bands=self.num_bands,
        )

        if self.num_bands is None:
            x = out
            self.y = None
        else:
            x, y = out
            self.y = torch.tensor(y, dtype=torch.long)

        # transform to tensor
        self.x = torch.tensor(x, dtype=torch.float32)

    @property
    def x_train(self) -> torch.Tensor:
        return self.x[: self.num_train]

    @property
    def x_val(self) -> torch.Tensor:
        return self.x[self.num_train : self.num_train + self.num_val]

    @property
    def x_test(self) -> torch.Tensor:
        return self.x[self.num_train + self.num_val :]

    @property
    def y_train(self) -> torch.Tensor | None:
        return None if self.y is None else self.y[: self.num_train]

    @property
    def y_val(self) -> torch.Tensor | None:
        return None if self.y is None else self.y[self.num_train : self.num_train + self.num_val]

    @property
    def y_test(self) -> torch.Tensor | None:
        return None if self.y is None else self.y[self.num_train + self.num_val :]

    def _make_dataset(self, x: torch.Tensor, y: torch.Tensor | None) -> TensorDataset:
        return TensorDataset(x) if y is None else TensorDataset(x, y)

    def setup(self, stage: str):
        """Set up train/test/val. datasets."""

        # create train/val. datasets
        if stage in ("fit", "validate"):
            self.train_set = self._make_dataset(self.x_train, self.y_train)
            self.val_set = self._make_dataset(self.x_val, self.y_val)

        # create test dataset
        elif stage == "test":
            self.test_set = self._make_dataset(self.x_test, self.y_test)

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create val. dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
        )
