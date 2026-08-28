import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

W1 = torch.tensor([
    [0.4, -0.2],
    [0.1, 0.6]
], requires_grad=True)

b1 = torch.tensor([
    [0.1],
    [-0.1]
], requires_grad=True)

W2 = torch.tensor([
    [0.3, -0.5]
], requires_grad=True)

b2 = torch.tensor([
    [0.2]
], requires_grad=True)

x = torch.tensor([
    [0.5],
    [-1.0]
], requires_grad=True)

y = torch.tensor([
    [1.0]
])

a = W1 @ x + b1
h = torch.sigmoid(a)

z = W2 @ h + b2
yhat = torch.sigmoid(z)

L = 0.5 * (yhat - y)**2

a.retain_grad()
h.retain_grad()
z.retain_grad()
yhat.retain_grad()

L.backward()

print("FORWARD")
print("a")
print(a)
print("h")
print(h)
print("z")
print(z)
print("yhat")
print(yhat)

print("L")
print(L)

print("L/yhat")
print(yhat.grad)
print("L/z")
print(z.grad)
print("L/h")
print(h.grad)
print("L/h")
print(h.grad)
print("L/b2")
print(b2.grad)
print("L/b1")
print(b1.grad)
print("L/W2")
print(W2.grad)
print("L/W1")
print(W1.grad)
print("L/x")
print(x.grad)
