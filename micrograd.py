#tiny autograd engine
#implements backpropagation 
#allows you to build out mathematical expression

import math
import numpy as np
import matplotlib.pyplot as plt
import os

#scalar value function
def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0))

xs = np.arange(-5, 5, 0.25)
#print(xs)
ys = f(xs)
#print(ys)
#plt.plot(xs, ys)
#plt.show()

#we're not going to write out the derivatives -- nobody can write out the expression for a neural net, it'd be 10s of thousands of terms!
#we want to really understand what the derivative is measuring and tells you about the function
h = 0.001
x = 3.0
print(f(x+h)) #do we expect f to increase when we very slightly increase x?
#yes, it gets very very slightly greater
print(f(x+h) - f(x)) #how positively did the function respond?
print((f(x+h) - f(x))/h) #the slope!

x = -3 #ok slope at 3 is 14; what's the slope at -3?
print((f(x+h) - f(x))/h) #the slope is about -22
x = 2/3
print((f(x+h) - f(x))/h) #at this point the slope is about 0


#let's get more complex
print("-----------")
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d, "\n")
#what is this derivative telling us?

#"get a bit tacky here", start at a very small value of h
h = 0.0001
#evaluate all inputs wrt h
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
print("d1:", d1) #4.0

a += h
d2 = a*b+c
print("d2:", d2) #3.9999...
#slightly less than 4. since we're making a slightly more positive, but b is still negative, we're adding *less* to d
#so the slope will be negative
print('slope:', (d2-d1)/h) #yep, -3.000...10772 etc
#differentiating d wrt a gives us b

#ok but what's the influence of b?
a = 2.0
b += h
d3 = a*b+c
print("d3:", d3) #we're making b slightly less negative, so my guess is the slope will be positive?

print('slope:', (d3-d1)/h) #yeah, it's 2.00, because differentiating d with respect to b gives us a

#ok but what's the influence of c?
b -= h 
c += h
d4 = a*b+c
print("d4:", d4) #we're making c slightly more positive so i think the slope will be positive but small

print('slope:', (d4-d1)/h) #0.9999...or 1. that's the rate at which d will increase as we scale c, so it's not much

#so here's what the derivative tells us about the function
#let's move to neural networks
#these are MASSIVE expressions. we need some data structures to hold those
print("-----------")

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
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            #take out's grad and propogate into self and other's grads
            #take local derivative times global derivative (deriv of final output of expression), wrt out.data
            self.grad += 1.0 * out.grad #in addition the local derivative is 1.0
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            #chain out.grad to self.grad
            #local derivative
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
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
        print(topo) #ordered our value object -- the last value is 0.707 which is our output, and all other nodes are laid before it
        #we're just calling ._backward on al objects in a topological order
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


    
a = Value(2.0, label='a')

b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

#we need to add addition to Value because this won't work
print(a.__add__(b))
print(a.__mul__(b))
print(a*b) 
#the above two lines mean the same thing!
#d = a*b+c

print(d._prev)

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

NUM_GRAPH = 0

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

    if not os.path.exists(f"graphs/graph_{NUM_GRAPH}.svg"):
        dot.render(f"graphs/graph_{NUM_GRAPH}")
    NUM_GRAPH += 1
    """else: 
        while os.path.exists(f"graphs/graph_{NUM_GRAPH}.svg"):
            NUM_GRAPH += 1
        dot.render(f"graphs/graph_{NUM_GRAPH}")"""

    return dot


f.grad = 4.0
d.grad = -2.0
L.grad = 1.0
dot = draw_dot(L)

#derivative of L wrt L -- if we change L by h, it changes by h


#L is the loss function, abcf are the weights of a neural net, the rest are the data
"""
we know L = d*f, so what is dL/dd? that should be f
how do we know that?
(f(x+h)-f(x))/h

dL = ((d+h)*f - d*f)/h
   = (df+hf - df)/h
   = (hf)/h
   = f
limit as h goes to 0 is f

"""
f.grad = 4.0
d.grad = -2.0
L.grad = 1.0
dot = draw_dot(L)

"""
this is the most important node; if you understand this gradient, you understand all of backpropogation and all of training of neural nets
dL / dc, deriv of L wrt C

we know dL/dd, but how is L sensitive to c/if we wiggle c how does that impact l through d?
we r aware of dd/dc
we should be able to put that together!

dd/dc = ?
d = c + e 
f(x+h)-f(x)/h
((c+h)+e - (c+e))/h
(c+h+e - c-e)/h
(h)/h
1 -> dd/dc = 1.0
by symmetry, dd/de = 1.0 also 

dd/dc is the local derivative of the little plus node that doesnt know anything about the rest of the graph, just that it is a plus node, and the local influence of c on d, and deriv of d wrt e. but we don't want the local derivative, we want derivative of L wrt c

chain rule!
dz/dx = dz/dy * dy/dz 
so
dL/dc = dL/dd * dd/dc
      =   f   *  1.0

a plus node literally just routes a gradient; the plus node local gradient is just 1, so by the chain rule it just routes that derivative forward to c and to e
"""
c.grad = -2.0
e.grad = -2.0

draw_dot(L)

NUM_GRAPH = 8

"""
dL/de = -2
now we want dL/da
chain rule -> dL/de * local gradient, de/da

recall e = a*b. what is de/da? if we differentiate wrt a,  we just get b!

dL/da = dL/de * de/da, we know both of those
dL/de is -2, de/da is just b, -3

dL/db = dL/de * de/db. so that's -2 again times the value of a
"""
a.grad = (-2.0 * -3.0)
b.grad = (-2.0 * 2.0)

draw_dot(L)

#verify

def lol(): 
    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    d = e+c; d.label='d'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    b.data += h
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    d = e+c; d.label='d'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data

    print((L2-L1)/h)

lol()
#this output was produced by some operation. we know the local derivatives and we can just always multiply those on and on
#backprop is just recursive application of the chain rule!

#nudge inputs to make L go up

#leaf nodes -- input values
for i in [a, b, c, f]:
    i.data += 0.01 * i.grad
#forward pass
e = a*b
d = e+c
L = d * f
#we expect a positive influence on L as a result of this -- become less negative
draw_dot(L)


print("-----neuron example------")
#squashing or activtion function applied to dot product of weights/inputs
plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)))
plt.grid()
#plt.show()

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

#n is the cellbody raw activation without the activation function (for now)
draw_dot(n)

#o is the output
o = n.__tanh__(); o.label = 'o'
#tanh is a hyperbolic function; so far weve only implemented + and *, and you can't make tanh out of that!
#ok so we added that

NUM_GRAPH=11
draw_dot(o)

"""
#start with backprop
# o = tanh
#d/dx tanh(x) = 1 - tanh(x)**2
#do/dn = 1 - tanh(n)**2 = 1 - o**2
print(o.data) #0.707
print(1-o.data**2) #0.5

n.grad = 0.5

#what is backprop going to do here? well + is a distributor of the gradient, so it's giong to flow back equally.
#so the two notes behind + behind n are going to get grad 0.5
x1w1x2w2.grad = 0.5
b.grad = 0.5

x1w1.grad = 0.5
x2w2.grad = 0.5

#pluses are so easy! if we want the output of this neuron to increase, then the influence of these expressions is positive

NUM_GRAPH=13
draw_dot(o)

#backproping thru the times node

x2.grad = w2.data * x2w2.grad 
w2.grad = x2.data * x2w2.grad

x1.grad = w1.data * x1w1.grad #local derivative times wrt w
w1.grad = x1.data * x1w1.grad

draw_dot(o)

#if we want the neuron's output to increase...
#w2 has no gradient and doesnt matter
#but w1 does have a gradient, and if it goes up, then the neuron's output goes up, and proportionally bc the gradient is 1"""

#wow that sure is a lot of work! lets NEVER DO IT MANUALLY AGAIN

NUM_GRAPH = 16
o.grad = 1.0 #initialize with 1
"""
o._backward() #populate the rest of the graph
draw_dot(o)

n._backward()
b._backward() #b doesn't have a backward! since b is a leaf node, by initialization this is an empty function
x1w1x2w2._backward() #the .5 gets further routed
x2w2._backward()
x1w1._backward()

draw_dot(o)
"""
#one last piece to get rid of -- calling _backward manually! we never want to call ._backward for any node before we've done everything after it. 
# we want to get its full dependencies to propogate to it before we continue backpropogation
# we can order these graphs using topological sort

NUM_GRAPH = 18
"""topo = []
visited = set()
def build_topo(v):
    #you will only be in the list once all of your children are in the list
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)
print(topo) #ordered our value object -- the last value is 0.707 which is our output, and all other nodes are laid before it
#we're just calling ._backward on al objects in a topological order

for node in reversed(topo):
    node._backward()"""

#now we have .backward as its own function
NUM_GRAPH = 19
o.backward()

draw_dot(o)

#we have a bad bug!!
a = Value(3.0, label='a')
b = a + a
b.label = 'b'
b.backward()
draw_dot(b)
#the forward pass works, but the gradient is incorrect
#that's because db/da should be 2 (just 1 + 1)

#every time we use a variable more than once
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b ; d.label='d'
e = a + b ; e.label = 'e'
f = d * e ; f.label = 'f'

f.backward()
draw_dot(f)
#ok now we've changed the backwards passes to be += -- accumulate instead of replace and that fixes the issue~!
draw_dot(f)

#ok now we go back to the x
