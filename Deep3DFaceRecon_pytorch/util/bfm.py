import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

class ParametricFaceModel(nn.Module):
    def __init__(self, bfm_folder='./BFM', recenter=True):
        super(ParametricFaceModel, self).__init__()
        self.bfm_folder = bfm_folder
        self.recenter = recenter
        
        # Use BFM 2019 model
        from .bfm2019_fixed import BFM2019Model
        model_path = f'{self.bfm_folder}/model2019_face12.h5'
        self.bfm2019 = BFM2019Model(model_path)
        
        # Copy relevant attributes from BFM2019 model
        self.mean_shape = self.bfm2019.mean_shape
        self.shape_basis = self.bfm2019.shape_basis
        self.mean_exp = self.bfm2019.mean_exp
        self.exp_basis = self.bfm2019.exp_basis
        self.mean_tex = self.bfm2019.mean_tex
        self.tex_basis = self.bfm2019.tex_basis
        self.face_buf = self.bfm2019.face_buf
        
        # Load or generate keypoints
        try:
            self.keypoints = torch.from_numpy(np.load(f'{self.bfm_folder}/keypoints_sim.npy'))
        except:
            # Use landmarks from BFM2019 if available
            self.keypoints = self.bfm2019.get_landmarks()
            if self.keypoints is None:
                # Fall back to default keypoints if needed
                self.keypoints = torch.arange(68)  # Placeholder, should be replaced with actual landmarks
        
        self.compute_mean_vertices()

    def compute_mean_vertices(self):
        n_vertices = self.mean_shape.shape[0] // 3
        self.mean_vertices = self.mean_shape.view(-1, 3)
        
        if self.recenter:
            mean_vertices = self.mean_vertices.clone()
            mean_vertices = mean_vertices.view(n_vertices, 3)
            mean_vertices = mean_vertices - torch.mean(mean_vertices, dim=0, keepdim=True)
            self.mean_vertices = mean_vertices.view(-1)

    def forward(self, shape_param, exp_param):
        """
        Forward pass of the parametric face model
        """
        batch_size = shape_param.shape[0]
        
        # Compute face shape
        vertices = self.mean_shape.clone()
        if shape_param is not None:
            vertices = vertices + torch.mm(shape_param, self.shape_basis)
        if exp_param is not None:
            vertices = vertices + torch.mm(exp_param, self.exp_basis)
            
        vertices = vertices.view(batch_size, -1, 3)
        
        return vertices
