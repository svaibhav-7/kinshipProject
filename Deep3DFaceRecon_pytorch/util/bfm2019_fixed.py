import h5py
import numpy as np
import torch
import torch.nn as nn
import os


class BFM2019Model(nn.Module):
    def __init__(self, bfm_h5_path, device='cuda'):
        super(BFM2019Model, self).__init__()
        self.device = device
        self._load_bfm2019(bfm_h5_path)

    def _load_bfm2019(self, h5_path):
        """Load BFM 2019 model from h5 file."""
        try:
            with h5py.File(h5_path, 'r') as f:
                print("Available root keys:", list(f.keys()))

                # Print hierarchy
                print("\nFile hierarchy:")
                f.visititems(lambda name, obj: print(name, type(obj)))

                # Helper function to safely read data
                def safe_read(path):
                    try:
                        if path in f:
                            dset = f[path]
                            if isinstance(dset, h5py.Dataset):
                                return np.array(dset)
                        return None
                    except Exception as e:
                        print(f"Error reading {path}: {str(e)}")
                        return None

                # Load shape components
                print("\nLoading shape components...")
                mean_shape = safe_read('shape/model/mean')
                shape_basis = safe_read('shape/model/pcaBasis')
                shape_variance = safe_read('shape/model/pcaVariance')

                if mean_shape is not None:
                    self.register_buffer('mean_shape', torch.tensor(mean_shape).float())
                if shape_basis is not None:
                    self.register_buffer('shape_basis', torch.tensor(shape_basis[:, :80]).float())
                if shape_variance is not None:
                    self.register_buffer('shape_std', torch.tensor(shape_variance[:80]).float().sqrt())

                # Load expression components
                print("\nLoading expression components...")
                mean_exp = safe_read('expression/model/mean')
                exp_basis = safe_read('expression/model/pcaBasis')
                exp_variance = safe_read('expression/model/pcaVariance')

                if mean_exp is not None:
                    self.register_buffer('mean_exp', torch.tensor(mean_exp).float())
                if exp_basis is not None:
                    self.register_buffer('exp_basis', torch.tensor(exp_basis[:, :64]).float())
                if exp_variance is not None:
                    self.register_buffer('exp_std', torch.tensor(exp_variance[:64]).float().sqrt())

                # Load color/texture components
                print("\nLoading texture components...")
                mean_tex = safe_read('color/model/mean')
                tex_basis = safe_read('color/model/pcaBasis')
                tex_variance = safe_read('color/model/pcaVariance')

                if mean_tex is not None:
                    self.register_buffer('mean_tex', torch.tensor(mean_tex).float())
                if tex_basis is not None:
                    self.register_buffer('tex_basis', torch.tensor(tex_basis[:, :80]).float())
                if tex_variance is not None:
                    self.register_buffer('tex_std', torch.tensor(tex_variance[:80]).float().sqrt())

                # Load triangle mesh indices
                print("\nLoading mesh topology...")
                cells = safe_read('representer/cells')
                if cells is None:
                    cells = safe_read('representer/triangles')
                if cells is None:
                    cells = safe_read('topology/cells')
                if cells is None:
                    cells = safe_read('topology/triangles')
                
                if cells is not None:
                    self.register_buffer('face_buf', torch.tensor(cells).long())
                else:
                    print("WARNING: Could not find mesh topology in h5 file.")
                    self._create_default_topology()

        except Exception as e:
            print(f"Error loading BFM 2019 model: {str(e)}")
            raise

    def _create_default_topology(self):
        """Create a default face topology for a simple face mesh.
        This creates a basic triangulated grid as a fallback."""
        print("Creating default topology...")
        # Create a simple grid mesh
        rows, cols = 32, 32  # Adjust these numbers as needed
        vertices_per_face = 3
        num_faces = (rows - 1) * (cols - 1) * 2
        faces = np.zeros((num_faces, vertices_per_face), dtype=np.int32)
        
        face_idx = 0
        for i in range(rows - 1):
            for j in range(cols - 1):
                # First triangle
                faces[face_idx] = [
                    i * cols + j,
                    (i + 1) * cols + j,
                    i * cols + j + 1
                ]
                face_idx += 1
                # Second triangle
                faces[face_idx] = [
                    (i + 1) * cols + j,
                    (i + 1) * cols + j + 1,
                    i * cols + j + 1
                ]
                face_idx += 1
        
        self.register_buffer('face_buf', torch.tensor(faces).long())

    def forward(self, shape_params=None, exp_params=None, tex_params=None):
        """
        Generate mesh vertices given shape and expression parameters.
        Args:
            shape_params: (B, 80) shape parameters
            exp_params: (B, 64) expression parameters
            tex_params: (B, 80) texture parameters
        Returns:
            vertices: (B, N, 3) mesh vertices
            texture: (B, N, 3) vertex colors if tex_params provided
        """
        batch_size = shape_params.shape[0] if shape_params is not None else 1
        
        # Initialize with mean shape
        vertices = self.mean_shape.clone().view(1, -1, 3).repeat(batch_size, 1, 1)

        # Add shape deviation
        if shape_params is not None:
            shape_basis = self.shape_basis.view(-1, 3, 80).permute(2, 0, 1)  # (80, N, 3)
            shape_deviation = torch.einsum('bl,lij->bij', shape_params * self.shape_std, shape_basis)
            vertices = vertices + shape_deviation

        # Add expression deviation
        if exp_params is not None:
            exp_basis = self.exp_basis.view(-1, 3, 64).permute(2, 0, 1)  # (64, N, 3)
            exp_deviation = torch.einsum('bl,lij->bij', exp_params * self.exp_std, exp_basis)
            vertices = vertices + exp_deviation

        # Generate texture if parameters provided
        texture = None
        if tex_params is not None:
            texture = self.mean_tex.clone().view(1, -1, 3).repeat(batch_size, 1, 1)
            tex_basis = self.tex_basis.view(-1, 3, 80).permute(2, 0, 1)  # (80, N, 3)
            tex_deviation = torch.einsum('bl,lij->bij', tex_params * self.tex_std, tex_basis)
            texture = texture + tex_deviation

        return vertices if texture is None else (vertices, texture)

    def get_faces(self):
        """Return the face (triangle) indices."""
        return self.face_buf

    def get_landmarks(self):
        """Return the landmark indices if available."""
        return self.landmarks if hasattr(self, 'landmarks') else None
