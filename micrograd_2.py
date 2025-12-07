#tiny autograd engine
#implements backpropagation 
#allows you to build out mathematical expression

import math
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_GRAPH = 0

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        #the derivative of L with respect to that value
        self.grad = 0.0 #assume every value does not affect the output; change of vartiable does not change loss function
        
        #our own little chain rule -- store how we chain output gradients to input gradients
        self._backward = lambda: None #by default it does nothing. for a leaf node, for example, there's nothing to do
        
        #children is a tuple but a set within the class for efficiency
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            #take out's grad and propogate into self and other's grads
            #take local derivative times global derivative (deriv of final output of expression), wrt out.data
            self.grad += 1.0 * out.grad #in addition the local derivative is 1.0
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            #chain out.grad to self.grad
            #local derivative
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting int/float powers 4 now'
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            #power rule: d/dx x^n = nx^(n-1)
            self.grad += other * self.data**(other - 1)
        
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other): #self - other
        return self + (-other)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    #implementing something more pwrful than divison, which division is a subcase of. 
    #a / b == a * (1/b) == a * (b**-1)
    
    #tanh = (e^2x - 1)/(e^2x + 1)
    def __tanh__(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh' )

        def _backward():
            #we have out.grad and we want to chain to self.grad
            #self.grad is the local of the operation we've done here - tanh
            #d/dx tanh(x) = 1 - tanh(x)**2
            self.grad += (1 - t**2) * out.grad #chained from the local gradient to self.grad

        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            #you will only be in the list once all of your children are in the list
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        #print(topo) #ordered our value object -- the last value is 0.707 which is our output, and all other nodes are laid before it
        #we're just calling ._backward on al objects in a topological order
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


#these expressions are about to get larger!
from graphviz import Digraph

def trace(root):
    #builds a set of all edges and nodes in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) #LR = left to right

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        #for any value in the graph create a rectangular record node for it)
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            #if this value is a result of some operation create a node for it
            dot.node(name=uid+n._op, label = n._op)
            #and connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        #connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    global NUM_GRAPH

    if not os.path.exists(f"graphs_2/graph_{NUM_GRAPH}.svg"):
        dot.render(f"graphs_2/graph_{NUM_GRAPH}")
    NUM_GRAPH += 1
    """else: 
        while os.path.exists(f"graphs/graph_{NUM_GRAPH}.svg"):
            NUM_GRAPH += 1
        dot.render(f"graphs/graph_{NUM_GRAPH}")"""

    return dot

a = Value(2.0)
print(2*a)
#a*2 will work, but 2*a will give an error!
#so we add rmul

b = Value(4.0)
print(a/b) #ok yay
print(b - a)

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
#b = Value(8, label='b')
b = Value(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

draw_dot(n)
#----
#o = n.__tanh__(); o.label = 'o'
#draw_dot(o)
NUM_GRAPH = 2
#we r changing how we implement o!
#tanh = (e^2x - 1)/(e^2x + 1)
e = (2*n).exp()
o = (e-1)/(e+1)
print(o)
draw_dot(o)

#using torch!
#micrograd has been used for scalars; torch works for multidimensional arrays called tensors
import torch
x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())

print(o.item())

import random
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self, x):
        #w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.__tanh__()
        #print(list(zip(self.w, x)))
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

        params = []
        for neuron in self.neurosn:
            ps = neuron.parameters()
            params.extend(ps)

        return params

    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        

x = [2.0,3.0]
#n = Neuron(2)
n = MLP(3, [4, 4, 1])
print(n(x))
#great, we have a single neuron! now let's make a layer of neurons
print(len(n.parameters()))
draw_dot(n(x))

#ok now we want to make a binary classifier neural net
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
print(ypred)
#we want all of these to match but currently they do not! how do we get the neural net to adjust?
#[Value(data=0.5353288204157638), Value(data=0.4145416203550441), Value(data=0.39519201444973384), Value(data=0.5370628263350736)]
#calculate a single number that measxures the performance of the neural net --  the loss
#we will implement a mean squared error loss

loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) #pair ground truths with predictions, zip iterates over tuples of them 
print("loss: ", loss)
#when yout = ground truth, prediction is target, you get 0. for now we are way off so loss is quite high

loss.backward()
print(n.layers[0].neurons[0].w[0].grad)
print(n.layers[0].neurons[0].w[0].data)

draw_dot(loss)

for p in n.parameters():
    p.data += -0.01 * p.grad #upd8 according to grad information
    #gradient -- vector pointing in the direction of increased loss. 
    # we want to update p.data in a small step size in the direction of the icnreased gradient
    # nope haha! we want to decrease and minimize loss. so we do negative. andrej you silly bastard...
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) #pair ground truths with predictions, zip iterates over tuples of them 
print("loss: ", loss) #it went slightly down from last time! waowee!

loss.backward()
for p in n.parameters():
    p.data += -0.01 * p.grad
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print("loss: ", loss) #it keeps going down!

for i in range(10):
    loss.backward()
    for p in n.parameters():
        p.data += -0.01 * p.grad
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    print("loss: ", loss)

print(ypred) #these are now quite close to our desired values!

#let's make this a little more respectable and implement an actual training loop
print("------------")
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
print(ypred)
for k in range(20):
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    #backward pass
    loss.backward()

    #update
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(k, loss.data)
print(ypred)