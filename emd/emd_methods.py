import torch
import numpy as np
import ot

def emd(M, weightX = [], weightY = []):
    M_np = M.clone().detach().cpu().numpy()
    if len(weightX) > 0 and len(weightY) > 0:
        weightX = weightX.tolist()
        weightY = weightY.tolist()
    flow = torch.as_tensor(ot.emd(weightX, weightY, M_np), device=M.device, dtype=M.dtype)
    return torch.sum(M * flow) # backward, treat flow matrix as a constant

def simple(P, x, y):
    comparison = (x.reshape(-1, 1) > y.reshape(1, -1)).astype(int)
    comparison_eq = 1 - (x.reshape(-1, 1) == y.reshape(1, -1)).astype(int)
    comparison[comparison == 0] = -1
    comparison *= comparison_eq
    return np.sum(P * comparison, axis=1)

class SlicedWasserstein_np(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, n_projections=10, p=2):
        # Convert tensors to NumPy arrays for calculations
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        
        d = X_np.shape[1]
        projections = ot.sliced.get_random_projections(d, n_projections, 0)
        
        X_projections = X_np.dot(projections)
        Y_projections = Y_np.dot(projections)
        
        sum_emd = 0
        flow_matrices = []
        
        for X_p, Y_p in zip(X_projections.T, Y_projections.T):
            emd_value, flow_matrix = ot.lp.emd2_1d(X_p, Y_p, log=True, metric='euclidean')
            sum_emd += emd_value
            flow_matrices.append(flow_matrix['G'])
        
        sum_emd /= n_projections
        ctx.save_for_backward(X, Y, torch.tensor(flow_matrices), torch.tensor(projections), torch.tensor(sum_emd), torch.tensor(p))
        sum_emd **= (1.0 / p)
        
        return (torch.tensor(sum_emd, dtype=torch.float32)).to(X.device) # Fixed return value

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, flow_matrices, projections, sum_emd, p = ctx.saved_tensors
        device = X.device
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        flow_matrices = flow_matrices.cpu().numpy()
        projections = projections.cpu().numpy().T
        sum_emd = sum_emd.item()
        p = p.item()
        
        grad_X = np.zeros_like(X)
        grad_Y = np.zeros_like(Y)
        
        for i in range(flow_matrices.shape[0]):
            flow_matrix = flow_matrices[i]
            X_p = X.dot(projections[i])
            Y_p = Y.dot(projections[i])
            df_dX = simple(flow_matrix, X_p, Y_p)
            df_dY = simple(flow_matrix.T, Y_p, X_p)
            
            grad_X += df_dX.reshape(-1, 1).dot(projections[i].reshape(1, -1))
            grad_Y += df_dY.reshape(-1, 1).dot(projections[i].reshape(1, -1))
        
        grad_X /= flow_matrices.shape[0]
        grad_Y /= flow_matrices.shape[0]
        
        # apply chain rule for sum_emd ** (1.0 / p)
        chain_coeff = (1.0 / p) * (sum_emd ** ((1.0 / p) - 1))
        grad_X *= chain_coeff * grad_output.item()
        grad_Y *= chain_coeff * grad_output.item()

        return torch.tensor(grad_X, dtype=torch.float32).to(device), torch.tensor(grad_Y, dtype=torch.float32).to(device), None, None
    


class SK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, reg=10, numItermax=10000):
        m, n = M.shape
        a = np.full(m, 1 / m)
        b = np.full(n, 1 / n)
        M_np = M.detach().cpu().numpy()
        flow = torch.tensor(ot.sinkhorn(a, b, M_np, reg=reg, numItermax=numItermax)).to(M.device)
        emd = (M * flow).sum()

        ctx.save_for_backward(flow)

        return emd
    
    @staticmethod
    def backward(ctx, grad_output):
        flow, = ctx.saved_tensors
        grad_cost_matrix = flow * grad_output
        
        return grad_cost_matrix, None, None

def swd(X_s, X_t):
    return ot.sliced_wasserstein_distance(X_s, X_t, n_projections=10)


def sinkhorn(M, reg=10, numItermax=10000):
    m = M.shape[0]
    n = M.shape[1]
    a = np.full(m, 1 / m)
    b = np.full(n, 1 / n)
    M_np = M.clone().detach().cpu().numpy()
    flow = torch.as_tensor(ot.sinkhorn(a, b, M_np, reg=reg, numItermax=numItermax), device=M.device, dtype=M.dtype)
    emd = torch.sum(M * flow)
    return emd


def fft(X_s, X_t, weightX = [], weightY = []):
    if isinstance(X_s, np.ndarray):
        X_s = torch.as_tensor(X_s)
        X_t = torch.as_tensor(X_t)

    n, m = X_s.shape[0], X_t.shape[0]

    if len(weightX) == 0 and len(weightY) == 0:
        a = np.full(n, 1 / n)
        b = np.full(m, 1 / m)
    else:
        a = weightX.detach().cpu().numpy()
        b = weightY.detach().cpu().numpy()
    
    half_n = n // 2
    # treated as a constant for backward
    compressed_flow = np.zeros(2*(n+m))
    index_1 = np.zeros(2*(n+m))
    index_2 = np.zeros(2*(n+m))
    
    i, j, k = 0, 0, 0
    while i < half_n and j < m:
        min_val = min(a[i], b[j])
        
        compressed_flow[k] = min_val
        index_1[k] = i
        index_2[k] = j
        k += 1
        
        compressed_flow[k] = min_val
        index_1[k] = n - i - 1
        index_2[k] = m - j - 1
        k += 1
        
        a[i] -= min_val
        b[j] -= min_val
        
        if a[i] == 0:
            i += 1
        if b[j] == 0:
            j += 1

    if n % 2 == 1:
        i = half_n
        while j < m:
            min_val = min(a[i], b[j])
            
            compressed_flow[k] = min_val
            index_1[k] = i
            index_2[k] = j
            k += 1
            
            a[i] -= min_val
            b[j] -= min_val
            
            if a[i] == 0:
                break
            if b[j] == 0:
                j += 1

    index_1 = index_1[:k]
    index_2 = index_2[:k]
    compressed_flow = compressed_flow[:k]

    index_1 = torch.as_tensor(index_1,device=X_s.device,dtype=torch.int)
    index_2 = torch.as_tensor(index_2,device=X_t.device,dtype=torch.int)

    distances = torch.norm(X_s[index_1] - X_t[index_2], dim=1)
    compressed_flow = torch.as_tensor(compressed_flow, device=distances.device,dtype=distances.dtype)
    return torch.sum(distances * compressed_flow)