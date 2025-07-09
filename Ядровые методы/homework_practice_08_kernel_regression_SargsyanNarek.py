import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(
            self,
            lr=0.01,
            regularization=1.0,
            tolerance=1e-2,
            max_iter=1000,
            batch_size=64,
            kernel_scale=1.0,
    ):
        """
        :param lr: learning rate
        :param regularization: regularization coefficient
        :param tolerance: stopping criterion for square of euclidean norm of weight difference
        :param max_iter: stopping criterion for iterations
        :param batch_size: size of the batches used in gradient descent steps
        :param kernel_scale: length scale in RBF kernel formula
        """

        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        # self.loss_history: list[float] = []
        self.kernel = RBF(kernel_scale)

    def calc_loss(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """

        loss = np.sum(0.5 * (x @ w - y) ** 2) + np.sum((self.regularization / 2) * (w.T @ x @ w))
        return loss

        # raise NotImplementedError

    def calc_grad(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        Calculating gradient for x and y dataset
        :param x: features array
        :param y: targets array
        """
        I = np.eye(x.shape[1])
        grad = (1 / len(x)) * ((x.T @ (x @ w - y)) + self.regularization* I @ w)
        return grad

        # raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров с помощью градиентного спуска
        :param x: features array
        :param y: targets array
        :return: self
        """

        k = self.kernel(x)
        self.x_train = x
        w = np.zeros(k.shape[1])

        w_list = [w.copy()]
        batch_indices = np.random.choice(k.shape[0], size=self.batch_size, replace=False)
        for i in range(self.max_iter):
            grad = self.calc_grad(k[batch_indices], y[batch_indices], w)
            w -= self.lr * grad
            w_list.append(w.copy())

            if np.linalg.norm(grad) < self.tolerance:
                break

        self.w = w_list[-1]
        # raise NotImplementedError
        return self

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров через аналитическое решение
        :param x: features array
        :param y: targets array
        :return: self
        """

        k = self.kernel(x)
        self.x_train = x
        self.w = np.dot(np.linalg.inv(k + self.regularization), y)
        # raise NotImplementedError
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        assert hasattr(self, "w"), "Сперва обучи модель пж"
        k = self.kernel(x, self.x_train)

        return k @ self.w

        # raise NotImplementedError