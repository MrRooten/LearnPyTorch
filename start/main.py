import torch

# 创建张量
x = torch.arange(12)
print(x)

# 访问张量的形状
print(x.shape)

# 元素总数
print(x.numel())

# 改变形状
X = x.reshape(3,4)
print(X)

# 创建形状为(2,3,4)的张量
three_d = torch.zeros(2,3,4)
print(three_d)

# 运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print("x", x)
print("y", y)
print("x + y", x + y)
print("x - y", x - y)
print("x * y", x * y)
print("x / y", x / y)
print("x ** y", x ** y)

# 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("a", a)
print("b", b)

print("a + b", a + b)

# 梯度，反向传播
x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print(f"y: {y}")
print(x.grad)
print(x.grad == 4*x)

# 非标量变量的反向传播
x = torch.arange(4.0, requires_grad=True)
y = x ** 2
y.sum().backward()
print(f"x.grad: {x.grad}")
