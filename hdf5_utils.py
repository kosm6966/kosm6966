from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class HDF5Container:
    """Dynamically populates a class from an HDF5 file.
    Example:
        grid = HDF5Container("my_grid_file.h5")
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._load(filepath)

    def _load(self, filepath: str):
        with h5py.File(filepath, 'r') as f:
            self._extract(f, self)

    def _extract(self, node, target):
        # Attributes (metadata on the node)
        for key, val in node.attrs.items():
            setattr(target, key, val)

        # Datasets and groups
        if hasattr(node, 'items'):
            for key, item in node.items():
                clean_key = key.replace(' ', '_').replace('-', '_')  # make valid attr name

                if isinstance(item, h5py.Dataset):
                    setattr(target, clean_key, item[()])  # [()] reads into numpy array

                elif isinstance(item, h5py.Group):
                    # Create a sub-object for nested groups
                    sub = type(clean_key, (), {})()  # anonymous object
                    self._extract(item, sub)
                    setattr(target, clean_key, sub)


@dataclass
class HDF5Dataset:
    """Stores information about an HDF5 dataset."""

    path: str
    shape: tuple
    dtype: str
    size: int  # Total number of elements
    nbytes: int  # Memory size in bytes
    attributes: dict[str, Any] = field(default_factory=dict)
    compression: str | None = None
    chunks: tuple | None = None
    fillvalue: Any | None = None

    def __str__(self) -> str:
        return (
            f"Dataset(path='{self.path}', shape={self.shape}, "
            f'dtype={self.dtype}, size={self.size}, nbytes={self.nbytes})'
        )


@dataclass
class HDF5Group:
    """Stores information about an HDF5 group."""

    path: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Group(path='{self.path}')"


@dataclass
class HDF5FileInfo:
    """
    Complete information about an HDF5 file with all its groups and datasets.
    Useful for large files where you want to extract and cache metadata.
    """

    filename: str
    file_size: int  # File size in bytes
    root_attributes: dict[str, Any] = field(default_factory=dict)
    groups: list[HDF5Group] = field(default_factory=list)
    datasets: list[HDF5Dataset] = field(default_factory=list)

    @property
    def total_dataset_size(self) -> int:
        """Calculate total size of all datasets in bytes."""
        return sum(ds.nbytes for ds in self.datasets)

    @property
    def num_datasets(self) -> int:
        """Number of datasets in the file."""
        return len(self.datasets)

    @property
    def num_groups(self) -> int:
        """Number of groups in the file."""
        return len(self.groups)

    def get_dataset(self, path: str) -> HDF5Dataset | None:
        """Retrieve a specific dataset by path."""
        if path[0] == '/':
            path = path[1:]
        for ds in self.datasets:
            if ds.path == path:
                return ds
        return None

    def get_group(self, path: str) -> HDF5Group | None:
        """Retrieve a specific group by path."""
        for grp in self.groups:
            if grp.path == path:
                return grp
        return None

    def list_datasets(self) -> list[str]:
        """Return list of all dataset paths."""
        return [ds.path for ds in self.datasets]

    def list_groups(self) -> list[str]:
        """Return list of all group paths."""
        return [grp.path for grp in self.groups]

    def summary(self) -> str:
        """Generate a summary of the HDF5 file structure."""
        lines = [
            f'{"=" * 60}',
            f'HDF5 File: {self.filename}',
            f'File Size: {self._format_bytes(self.file_size)}',
            f'Total Dataset Size: {self._format_bytes(self.total_dataset_size)}',
            f'Number of Groups: {self.num_groups}',
            f'Number of Datasets: {self.num_datasets}',
            f'{"=" * 60}',
        ]

        if self.root_attributes:
            lines.append('Root Attributes:')
            for name, value in self.root_attributes.items():
                lines.append(f'  - {name}: {self._format_value(value)}')

        if self.groups:
            lines.append('\nGroups:')
            for group in sorted(self.groups, key=lambda g: g.path):
                lines.append(f'  /{group.path}')
                if group.attributes:
                    for name, value in group.attributes.items():
                        lines.append(f'    - {name}: {self._format_value(value)}')

        if self.datasets:
            lines.append('\nDatasets:')
            for dataset in sorted(self.datasets, key=lambda d: d.path):
                lines.append(f'  /{dataset.path}')
                lines.append(f'    Shape: {dataset.shape}, Dtype: {dataset.dtype}')
                lines.append(f'    Size: {self._format_bytes(dataset.nbytes)}')
                if dataset.compression:
                    lines.append(f'    Compression: {dataset.compression}')
                if dataset.chunks:
                    lines.append(f'    Chunks: {dataset.chunks}')
                if dataset.attributes:
                    for name, value in dataset.attributes.items():
                        lines.append(f'    - {name}: {self._format_value(value)}')

        lines.append(f'{"=" * 60}')
        return '\n'.join(lines)

    @staticmethod
    def _format_bytes(nbytes: int) -> str:
        """Format bytes in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if nbytes < 1024.0:
                return f'{nbytes:.2f} {unit}'
            nbytes /= 1024.0
        return f'{nbytes:.2f} PB'

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format attribute values for display."""
        if isinstance(value, np.ndarray):
            return np.array2string(value, max_line_width=40)
        return str(value)


class HDF5Analyzer:
    """
    Analyzer class to extract and store HDF5 file information.
    Designed to handle large files efficiently.
    """

    def __init__(self):
        self.file_info = None
        self._filename = None
        self._root = None

    def analyze(self, filename: str) -> HDF5FileInfo:
        """
        Analyze an HDF5 file and store all metadata.

        Args:
            filename: Path to the HDF5 file

        Returns:
            HDF5FileInfo object containing all extracted information
        """
        try:
            # Get file size
            file_size = Path(filename).stat().st_size

            file_info = HDF5FileInfo(filename=filename, file_size=file_size)

            self._filename = filename

            self._root = filename.rpartition('/')[0]

            # Open and traverse the HDF5 file
            with h5py.File(filename, 'r') as f:
                # Store root attributes
                file_info.root_attributes = self._extract_attributes(f)

                # Recursively process all items
                f.visititems(lambda name, obj: self._process_item(name, obj, file_info))

            self.file_info = file_info
            return file_info

        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{filename}' was not found.") from None
        except Exception as e:
            raise RuntimeError(f"An error occurred while analyzing the HDF5 file '{filename}': {type(e).__name__}: {e}") from e


    def _process_item(self, name: str, obj: Any, file_info: HDF5FileInfo) -> None:
        """Process an individual HDF5 item (group or dataset)."""
        attributes = self._extract_attributes(obj)

        if isinstance(obj, h5py.Group):
            group = HDF5Group(path=name, attributes=attributes)
            file_info.groups.append(group)

        elif isinstance(obj, h5py.Dataset):
            dataset = HDF5Dataset(
                path=name,
                shape=obj.shape,
                dtype=str(obj.dtype),
                size=obj.size,
                nbytes=obj.nbytes,
                attributes=attributes,
                compression=obj.compression,
                chunks=obj.chunks,
                fillvalue=obj.fillvalue,
            )
            file_info.datasets.append(dataset)

    @staticmethod
    def _extract_attributes(obj: Any) -> dict[str, Any]:
        """Extract and format attributes from an HDF5 object."""
        attributes = {}
        if obj.attrs:
            for attr_name, attr_value in obj.attrs.items():
                # Convert numpy arrays to lists for better serialization
                if isinstance(attr_value, np.ndarray):
                    attributes[attr_name] = attr_value.tolist()
                    if attributes[attr_name] and isinstance(attributes[attr_name][0], bytes):
                        attributes[attr_name] = [val.decode() for val in attributes[attr_name]]
                elif isinstance(attr_value, bytes):
                    attributes[attr_name] = attr_value.decode()
                else:
                    attributes[attr_name] = attr_value
        return attributes

    def extract_dataset(self, dataset_path: str, slicing: tuple | None = None) -> np.ndarray | Any:
        """
        Extract dataset from the HDF5 file.

        Args:
            dataset_path: Path to the dataset (e.g., 'group1/subgroup/dataset1')
            slicing: Optional tuple for slicing (e.g., (0, 100) for first 100 rows)
                     Can use Python's slice notation (0:100:2 becomes slice(0,100,2))

        Returns:
            The dataset as a numpy array or scalar value

        Raises:
            ValueError: If dataset not found or invalid slicing
            Exception: If error reading the file
        """
        if not self._filename:
            raise ValueError('No file has been analyzed yet. Call analyze() first.')

        try:
            with h5py.File(self._filename, 'r') as f:
                if dataset_path not in f:
                    raise ValueError(
                        f"Dataset '{dataset_path}' not found. Available datasets: {self.file_info.list_datasets()}"
                    )

                dataset = f[dataset_path]

                # Extract data with optional slicing
                if slicing is not None:
                    data = dataset[slicing]
                else:
                    data = dataset[()]  # Load entire dataset

                return data

        except Exception as e:
            raise RuntimeError(f"Error extracting dataset '{dataset_path}': {e}") from e

    def extract_attribute(self, target_path: str, attribute_name: str) -> Any:
        """
        Extract an attribute from a group or dataset.

        Args:
            target_path: Path to the group or dataset (e.g., 'group1/dataset1')
                        Use empty string '' for root attributes
            attribute_name: Name of the attribute

        Returns:
            The attribute value

        Raises:
            ValueError: If target not found or attribute doesn't exist
            Exception: If error reading the file
        """
        if not self._filename:
            raise ValueError('No file has been analyzed yet. Call analyze() first.')

        try:
            with h5py.File(self._filename, 'r') as f:
                # Handle root attributes
                if target_path == '' or target_path == '/':
                    obj = f
                else:
                    if target_path not in f:
                        raise ValueError(f"Path '{target_path}' not found in the HDF5 file.")
                    obj = f[target_path]

                if attribute_name not in obj.attrs:
                    available = list(obj.attrs.keys())
                    raise ValueError(
                        f"Attribute '{attribute_name}' not found at '{target_path}'. Available attributes: {available}"
                    )

                attr_value = obj.attrs[attribute_name]

                # Convert numpy arrays to lists for better handling
                if isinstance(attr_value, np.ndarray):
                    return attr_value.tolist()
                elif isinstance(attr_value, bytes):
                    return attr_value.decode()
                return attr_value

        except Exception as e:
            raise RuntimeError(f"Error extracting attribute '{attribute_name}' from '{target_path}': {e}") from e

    def extract_group_attributes(self, group_path: str = '') -> dict[str, Any]:
        """
        Extract all attributes from a group or the root.

        Args:
            group_path: Path to the group (empty string '' for root)

        Returns:
            Dictionary of all attributes

        Raises:
            ValueError: If path not found
            Exception: If error reading the file
        """
        if not self._filename:
            raise ValueError('No file has been analyzed yet. Call analyze() first.')

        try:
            with h5py.File(self._filename, 'r') as f:
                if group_path == '' or group_path == '/':
                    obj = f
                else:
                    if group_path not in f:
                        raise ValueError(f"Path '{group_path}' not found in the HDF5 file.")
                    obj = f[group_path]

                return self._extract_attributes(obj)

        except Exception as e:
            raise RuntimeError(f"Error extracting attributes from '{group_path}': {e}") from e

    def save_dataset_to_file(
        self, dataset_path: str, output_file: str, slicing: tuple | None = None, format: str = 'npy'
    ) -> None:
        """
        Extract a dataset and save it to a file.

        Args:
            dataset_path: Path to the dataset in the HDF5 file
            output_file: Path to save the output file
            slicing: Optional slicing tuple
            format: Output format ('npy' for numpy, 'csv' for CSV)

        Raises:
            ValueError: If format not supported
            Exception: If error during extraction or saving
        """
        try:
            data = self.extract_dataset(dataset_path, slicing)

            if format.lower() == 'npy':
                np.save(output_file, data)
            elif format.lower() == 'csv':
                if data.ndim > 2:
                    raise ValueError('CSV format only supports 1D or 2D arrays')
                np.savetxt(output_file, data, delimiter=',')
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'npy' or 'csv'")

            print(f'Dataset saved to {output_file}')

        except Exception as e:
            raise IOError(f'Error saving dataset: {e}') from e

    def get_dataset_info(self, dataset_path: str) -> str:
        """
        Get formatted information about a specific dataset.

        Args:
            dataset_path: Path to the dataset

        Returns:
            Formatted string with dataset information
        """
        ds = self.file_info.get_dataset(dataset_path)
        if not ds:
            return f"Dataset '{dataset_path}' not found."

        lines = [
            f'Dataset: /{ds.path}',
            f'  Shape: {ds.shape}',
            f'  Dtype: {ds.dtype}',
            f'  Size: {self.file_info._format_bytes(ds.nbytes)}',
            f'  Elements: {ds.size:,}',
        ]

        if ds.compression:
            lines.append(f'  Compression: {ds.compression}')
        if ds.chunks:
            lines.append(f'  Chunks: {ds.chunks}')
        if ds.fillvalue is not None:
            lines.append(f'  Fill Value: {ds.fillvalue}')

        if ds.attributes:
            lines.append('  Attributes:')
            for name, value in ds.attributes.items():
                lines.append(f'    - {name}: {self.file_info._format_value(value)}')

        return '\n'.join(lines)
