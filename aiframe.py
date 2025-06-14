import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from mindspore.nn import Cell, Adam, MSELoss, TrainOneStepCell
from mindspore.common.initializer import XavierNormal
from mindspore import numpy as mnp
from scipy.special import gamma
from sciai.architecture import MLP
import matplotlib.pyplot as plt
from mpmath import binomial
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

alpha = 1.8

def random_points_disk(num_points, dim, center, radius):
    if dim != 2:
        raise ValueError("目前只支持2维圆盘采样。")

    theta = np.random.rand(num_points) * 2 * np.pi
    r = radius * np.sqrt(np.random.rand(num_points))

    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    sampled_points = np.stack((x, y), axis=1)
    return sampled_points

def boundary_points_disk(num_points, dim, center, radius):
    if dim != 2:
        raise ValueError("目前只支持2维边界采样。")

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]

    sampled_points = np.stack((x, y), axis=1)
    return sampled_points

def func(x, alpha_val):
    if isinstance(x, Tensor):
        x_np = x.asnumpy()
    else:
        x_np = x

    return (np.abs(1 - np.linalg.norm(x_np, axis=1, keepdims=True) ** 2)) ** (
        1 + alpha_val / 2
    )

class Net(Cell):
    def __init__(self, layers, activation, weight_init):
        super(Net, self).__init__()
        self.mlp = MLP(layers, activation=activation, weight_init=weight_init)

    def construct(self, x):
        return self.mlp(x)

class FractionalPoissonLoss(Cell):
    def __init__(self, net, alpha_val, num_directions=8, gl_h_factor=0.01):
        super(FractionalPoissonLoss, self).__init__()
        self.net = net
        self.alpha = alpha_val
        self.num_directions = num_directions
        self.gl_h_factor = gl_h_factor

        self.C_alpha_D = gamma((1 - self.alpha) / 2) * gamma((2 + self.alpha) / 2) / (2 * np.pi ** 1.5)

        self.mse_loss = MSELoss()

        # 均匀划分
        # thetas = np.linspace(0, 2 * np.pi, self.num_directions, endpoint=False)
        # self.directions = Tensor(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), ms.float32)
        # self.direction_weights = Tensor(np.ones(self.num_directions) * (2 * np.pi / self.num_directions), ms.float32)

        # 高斯-勒让德函数划分
        gauss_x, gauss_w = np.polynomial.legendre.leggauss(self.num_directions)
        thetas = np.pi * gauss_x + np.pi
        self.directions = Tensor(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), ms.float32)
        self.direction_weights = Tensor(np.pi * gauss_w, ms.float32)

        self.gl_coeffs_np = []
        self.gl_max_k = 200
        self.gl_coeffs_np = []
        for k in range(self.gl_max_k):
            coeff = (-1)**k * binomial(self.alpha, k)
            self.gl_coeffs_np.append(float(coeff))  # 可转为 float 或保留 mpf 类型
        self.gl_coeffs = Tensor(np.array(self.gl_coeffs_np), ms.float32)

    def _compute_directional_fractional_derivative_batch(self, u_net_func, x_points_tensor, directions_tensor, h):
        # x_points_tensor: (num_domain, dim)
        # directions_tensor: (num_directions, dim)
        # h: scalar

        num_domain = x_points_tensor.shape[0]
        num_directions = directions_tensor.shape[0]

        # Reshape for broadcasting
        # x_points_tensor: (num_domain, 1, 1, dim)
        x_points_expanded = ops.expand_dims(ops.expand_dims(x_points_tensor, axis=1), axis=1)

        # directions_tensor: (1, num_directions, 1, dim)
        directions_expanded = ops.expand_dims(ops.expand_dims(directions_tensor, axis=0), axis=2)

        # k_values: (1, 1, gl_max_k, 1)
        k_values = mnp.arange(self.gl_max_k, dtype=ms.float32).reshape(1, 1, self.gl_max_k, 1)


        # shifted_points: (num_domain, num_directions, gl_max_k, dim)
        shifted_points = x_points_expanded - ops.mul(k_values * h, directions_expanded)


        # Flatten shifted_points to pass through the network: (num_domain * num_directions * gl_max_k, dim)
        shifted_points_flat = shifted_points.view(-1, shifted_points.shape[-1])

        y_raw_shifted_flat = u_net_func(shifted_points_flat)

        # Reshape back: (num_domain, num_directions, gl_max_k, 1)
        y_raw_shifted = y_raw_shifted_flat.view(num_domain, num_directions, self.gl_max_k, 1)

        shifted_points_norm_sq = ops.sum(shifted_points ** 2, dim=-1, keepdim=True)

        u_transformed_potential = ops.mul((1 - shifted_points_norm_sq), y_raw_shifted)
        is_inside_or_boundary = ops.less_equal(shifted_points_norm_sq, 1.0 + 1e-7)
        u_shifted = ops.where(is_inside_or_boundary, u_transformed_potential, ops.zeros_like(u_transformed_potential))

        # gl_coeffs: (gl_max_k, 1) -> (1, 1, gl_max_k, 1) for broadcasting
        gl_coeffs_expanded = ops.expand_dims(ops.expand_dims(ops.expand_dims(self.gl_coeffs, axis=0), axis=0), axis=3)

        # weighted_u_shifted: (num_domain, num_directions, gl_max_k, 1)
        weighted_u_shifted = ops.mul(gl_coeffs_expanded, u_shifted)

        # gl_sum: (num_domain, num_directions, 1)
        gl_sum = ops.sum(weighted_u_shifted, dim=2)

        # directional_deriv: (num_domain, num_directions, 1)
        directional_deriv = ops.mul(gl_sum, (1.0 / (h ** self.alpha)))

        return directional_deriv

    def construct(self, x_domain, x_boundary):
        h_gl = self.gl_h_factor

        # Vectorized computation of directional fractional derivatives
        # L_alpha_u_directional: (num_domain, num_directions, 1)
        L_alpha_u_directional = self._compute_directional_fractional_derivative_batch(
            self.net, x_domain, self.directions, h_gl
        )

        # Weighted sum over directions
        # direction_weights: (num_directions,) -> (1, num_directions, 1) for broadcasting
        direction_weights_expanded = ops.expand_dims(ops.expand_dims(self.direction_weights, axis=0), axis=2)

        # directional_deriv_sum: (num_domain, 1)
        directional_deriv_sum = ops.sum(ops.mul(direction_weights_expanded, L_alpha_u_directional), dim=1)

        L_alpha_u = ops.mul(self.C_alpha_D, directional_deriv_sum) # C_alpha_D is now a Tensor

        lhs = L_alpha_u[:, 0]

        # Convert the scalar constant derived from gamma functions to a MindSpore Tensor
        rhs_constant = Tensor(2 ** self.alpha
                              * gamma(2 + self.alpha / 2)
                              * gamma(1 + self.alpha / 2), ms.float32)

        rhs = (
            rhs_constant
            * (1 - (1 + self.alpha / 2) * ops.sum(x_domain ** 2, dim=1))
        )
        pde_loss = self.mse_loss(lhs, rhs)
        
        y_raw_boundary = self.net(x_boundary)
        u_pred_boundary = ops.mul((1 - ops.sum(x_boundary ** 2, dim=1, keepdim=True)), y_raw_boundary)
        bc_loss = self.mse_loss(u_pred_boundary, Tensor(func(x_boundary, self.alpha), ms.float32))
        return pde_loss + bc_loss

# --- 5. 主训练函数 (保持不变) ---
def train(load_ckpt):
    layers = [2] + [20] * 4 + [1]
    activation = ops.Tanh()
    weight_init = XavierNormal()
    net = Net(layers, activation, weight_init)
    load_ckpt_path = "./ckpt/fractional_poisson.ckpt"
    if load_ckpt == False:
        num_domain = 100
        num_boundary = 100

        x_domain_np = random_points_disk(num_domain, 2, center=[0, 0], radius=1)
        x_domain = Tensor(x_domain_np, ms.float32)

        x_boundary_np = boundary_points_disk(num_boundary, 2, center=[0, 0], radius=1)
        x_boundary = Tensor(x_boundary_np, ms.float32)

        # Consider increasing gl_h_factor slightly if initial loss is NaN due to very small h
        # Or, adjust it based on your problem's scaling.
        loss_fn = FractionalPoissonLoss(net, alpha, num_directions=8, gl_h_factor=0.01)
        optimizer = Adam(net.trainable_params(), learning_rate=1e-3)

        train_net = TrainOneStepCell(loss_fn, optimizer)
        train_net.set_train()

        print("Start Training")
        iterations = 20000
        loss_list = []
        for i in range(iterations):
            loss = train_net(x_domain, x_boundary)
            if (i+1) % 2000 == 0:
                loss_list.append(loss.asnumpy())
                print(f"迭代 {i+1}, 损失: {loss.asnumpy():.6f}")
        save_checkpoint(net, load_ckpt_path)
        print("Train End")

        # loss曲线画图
        # 每2000步记录一次，横坐标是 step
        steps = list(range(2000, iterations+1, 2000))
        plt.figure(figsize=(8, 5))
        plt.plot(steps, loss_list, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve (Every 2000 Iterations)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve.png")  # 保存为图片
        plt.show()
    else:
        param_dict = load_checkpoint(load_ckpt_path)
        load_param_into_net(net, param_dict)
        print("Checkpoint loaded.")

    # --- 评估和绘图 ---
    print("Start Test and Plotting")
    # 创建一个网格点用于可视化，而不是随机点
    grid_points = 100
    x_coords = np.linspace(-1, 1, grid_points)
    y_coords = np.linspace(-1, 1, grid_points)
    X_plot, Y_plot = np.meshgrid(x_coords, y_coords)
    
    # 筛选出圆盘内的点
    R_sq = X_plot**2 + Y_plot**2
    mask = R_sq <= 1.0 # 过滤掉圆盘外的点
    
    X_test_plot_np = np.stack([X_plot[mask], Y_plot[mask]], axis=1)
    X_test_plot_tensor = Tensor(X_test_plot_np, ms.float32)

    net.set_train(False)
    y_raw_pred_test = net(X_test_plot_tensor).asnumpy()
    y_pred_transformed = (1 - np.sum(X_test_plot_np ** 2, axis=1, keepdims=True)) * y_raw_pred_test
    
    y_true = func(X_test_plot_np, alpha)

    # --- 绘图部分 ---
    fig = plt.figure(figsize=(15, 6))

    # 绘制预测解
    ax1 = fig.add_subplot(121, projection='3d')
    # 将预测结果填充回网格
    Z_pred = np.zeros_like(R_sq)
    Z_pred[mask] = y_pred_transformed.flatten()
    ax1.plot_surface(X_plot, Y_plot, Z_pred, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Predicted Solution for alpha={alpha}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u(x,y)')

    # 绘制精确解
    ax2 = fig.add_subplot(122, projection='3d')
    # 将精确结果填充回网格
    Z_true = np.zeros_like(R_sq)
    Z_true[mask] = y_true.flatten()
    ax2.plot_surface(X_plot, Y_plot, Z_true, cmap='viridis', edgecolor='none')
    ax2.set_title(f'Exact Solution for alpha={alpha}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u(x,y)')

    plt.tight_layout()
    plt.show()
    plt.savefig('compare.png')
    print("Plotting complete.")
if __name__ == "__main__":
    load_ckpt = True
    train(load_ckpt)