import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy as sp
import networkx as nx
import pylab

izhikevich_a = 0.02
izhikevich_b = 0.2
izhikevich_c = -65
izhikevich_d = 8

#izhikevich_threshold = 30-izhikevich_c
izhikevich_threshold = -izhikevich_c

connectivity = 0.3
neuron_count = 20

class Neuron:
	def __init__(self):
		self.v = izhikevich_c
		self.u = izhikevich_b * self.v
		self.c = 0
	
	def fired(self):
		if(self.v >= 30):
			self.v = izhikevich_c
			self.u = self.u + izhikevich_d
			return True
		return False
	
	def halfstep(self):
		self.v = self.v + (0.04 * self.v**2 + 5*self.v + 140 - self.u)  / 2
		self.u = self.u + (izhikevich_a*(izhikevich_b*self.v - self.u)) / 2
		self.c = self.c - (0.001*self.c)/2
		return self.fired()
	
	def ms_step(self):
		r1 = self.halfstep()
		r2 = self.halfstep()
		#spontaneous = False# np.random.random_sample()<0.0001#5
		#if(spontaneous):
		#	print('beep')
		return r1 or r2# or spontaneous
		
	def input(self, value):
		self.v = self.v + value
		#self.c = self.c*0.95 + value*0.05
		self.c = self.c + value*0.1
		return izhikevich_threshold - self.c

synapse_tRelax = 100
synapse_Utilisation = 0.4
		
class Synapse:
	def __init__(self):
		self.w = izhikevich_threshold*connectivity
		#inactive transmitter
		self.I = 0
		self.t_last = 0
		
	def effect(self, t):
		dt = t - self.t_last
		self.t_last = t
		#reuptake
		self.I = self.I * np.exp(-dt/synapse_tRelax)
		
		#releaseable
		#R = 1-I
		
		#effect
		E = (1-self.I)*synapse_Utilisation
		self.I = self.I + E
		return self.w*E
		
	def ms_step(self):
		pass
	
	def learn(self, modulator):
		if(self.w < 0):
			return
		self.w = self.w + 0.01*modulator
		if(self.w < 0):
			self.w = 0
		#print(0.01*modulator, self.w)


class SpikeDelay:
	def __init__(self, size, default):
		self.index = 0
		self.size = size
		self._data = [default for _ in range(size)]
	
	def current(self):
		return self._data[self.index]
		
	def set(self, value):
		self._data[self.index] = value
		self.index = (self.index+1) % self.size

		
def initPartial():		
	net = nx.DiGraph()
	for idx in range(neuron_count):
		net.add_node(Neuron())
		#net.add_node(idx, neuron=Neuron())


	target_degree = int( neuron_count * connectivity )
	for _idx, neuron in enumerate(net.nodes):
		needs_connections = True
		while needs_connections:
			targets = [n for n in net.nodes if (net.in_degree[n] < target_degree and not net.has_edge(neuron, n) and n is not neuron)]
			if ( len(targets) == 0 ):
				break
			idx_to = np.random.randint(len(targets))	
			net.add_edge(neuron, targets[idx_to], object=Synapse())
			needs_connections = not net.out_degree[neuron] is target_degree
			#print(_idx, idx_to, net.out_degree[neuron], target_degree, needs_connections, len(targets), net.in_degree[targets[idx_to]])
	return net

		
def initComplete():
	net = nx.DiGraph()
	for idx in range(neuron_count):
		net.add_node(Neuron())

	for source in net.nodes:
		for target in net.nodes:
			net.add_edge(source, target, object=Synapse())
	return net

#net = initComplete()
net = initPartial()

spikes = []
voltage = []
calcium = []
for neuron in net.nodes:
	spikes.append([])
	voltage.append([])
	calcium.append([])
	
axons = [SpikeDelay(np.random.randint(5,15), False) for _ in net.nodes]

training = 00000
steps = 70000

time = 0
for t in range(training):
	list(net)[0].input(5)
	
	for idxNeuron, neuron in enumerate(net.nodes):
		transmission = axons[idxNeuron].current()
		axons[idxNeuron].set(neuron.ms_step())
		if (transmission):
			for edge in net.out_edges(neuron, data=True):
				synapse = edge[2]['object']
				ca = edge[1].input(synapse.effect(time+t))
				synapse.learn(ca)
time = time+training

for t in range(steps):
	#pulse = 5 - (t/10000)
	#if(pulse > 0):
	#	list(net)[0].input(pulse)
	list(net)[0].input(5)
	
	for idxNeuron, neuron in enumerate(net.nodes):
		voltage[idxNeuron].append(neuron.v)
		
		transmission = axons[idxNeuron].current()
		spiked = neuron.ms_step()
		axons[idxNeuron].set(spiked)
		if (spiked):
			spikes[idxNeuron].append(t)
			
		if (transmission):
			for edge in net.out_edges(neuron, data=True):
				synapse = edge[2]['object']
				ca = edge[1].input(synapse.effect(time+t))
				synapse.learn(ca)
		voltage[idxNeuron].append(neuron.v)
		calcium[idxNeuron].append(neuron.c)
#print (spikes)
print('calc done')

#plt.figure(1)
#plt.title('Topology plot')
#nx.draw(net, with_labels = False)

#plt.figure(2)
#plt.eventplot(spikes, linelengths = [0.5 for n in net.nodes])     
#plt.title('Spike raster plot')
#plt.xlabel('Neuron')
#plt.ylabel('Spike')

fig, axs = plt.subplots(len(voltage), 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

plt.title('Membrane potential plot')
plt.xlabel('Neuron')
plt.ylabel('v')
for idx in range(neuron_count):
	plt.eventplot(spikes, linelengths = [0.5 for n in net.nodes])    
	axs[idx].plot(np.arange(2*steps)/2, voltage[idx], 'xkcd:blue')
	axs[idx].plot(np.arange(steps), calcium[idx], 'xkcd:stone')
	axs[idx].eventplot(spikes[idx], linelengths=[150], colors=['xkcd:red'])
	axs[idx].set_ylim(-100,100)
plt.show()
