import torch
from model.dynamics import Dynamic
from utils import block_diag
from model.components import GMM2D


class SingleIntegrator(Dynamic):
    def init_constants(self):
        self.F = torch.eye(4, device=self.device, dtype=torch.float32)
        self.F[0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * self.dt
        self.F_t = self.F.transpose(-2, -1)

    def integrate_samples(self, v, x=None):
        """
        Integrates deterministic samples of velocity.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        """
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        return torch.cumsum(v, dim=2) * self.dt + p_0

    def integrate_distribution(self, v_dist, x=None):
        r"""
        Integrates the GMM velocity distribution to a distribution over position.
        The Kalman Equations are used.

        .. math:: \mu_{t+1} =\textbf{F} \mu_{t}

        .. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

        .. math::
            \textbf{F} = \left[
                            \begin{array}{cccc}
                                \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                                \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                                0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                                0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                            \end{array}
                        \right]_{t}

        :param v_dist: Joint GMM Distribution over velocity in x and y direction.
        :param x: Not used for SI.
        :return: Joint GMM Distribution over position in x and y direction.
        """
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        pos_dist_sigma_matrix_list = []

        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * self.dt

        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(sample_batch_dim + [v_dist.components, 2, 2], device=self.device)

        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = block_diag([pos_dist_sigma_matrix_t, vel_sigma_matrix_t])
            pos_dist_sigma_matrix_t = self.F[..., :2, :].matmul(full_sigma_matrix_t.matmul(self.F_t)[..., :2])
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        #print('行人')
        #return GMM2D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_dist_sigma_matrix)
        #############添加
        corrs_sigma12 = pos_dist_sigma_matrix[..., 0, 1]
        sigma_1 = torch.clamp(pos_dist_sigma_matrix[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(pos_dist_sigma_matrix[..., 1, 1], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs = corrs_sigma12 / (torch.prod(sigmas, dim=-1))  
        
        log_pis = torch.clamp(v_dist.log_pis, min=-1e5)
        log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        pos_mus = self.reshape_to_components(pos_mus)         # [..., N, 2]####这行命令没有改变原来变量的形状
        # print('pos_mus均值位置2',pos_mus.shape)
        log_sigmas = self.reshape_to_components(log_sigmas) 
        sigmas = torch.exp(log_sigmas) 

        return GMM2D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_dist_sigma_matrix),pos_mus,sigmas,corrs,log_pis

    def reshape_to_components(self, tensor):
        #print('张量类型',tensor.shape,len(tensor.shape))
        if len(tensor.shape) == 5:
            #print('运行1',tensor.shape)
            return tensor
        else:
            #print('运行2',torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions]).shape)
            return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])