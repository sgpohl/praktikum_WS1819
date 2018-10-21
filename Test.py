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
izhikevich_threshold = -izhikevich_c*0.6

connectivity = 0.3
neuron_count = 20

global_reservoir = 1

np.random.seed()

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

synapse_tRelax = 500
synapse_Utilisation = 0.6
		
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
		#print(E)
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

def insertInhibition(net, percentage):
	counter = 0
	edges = [s for (_,_,s) in (net.edges.data('object'))]
	np.random.shuffle(edges)
	for synapse in edges:# net.out_edges(data=True):
		counter = counter-percentage
		if(counter < 0):
			counter = counter+1
			synapse.w = -5*synapse.w
	return net
	
def insertInhibitionNeurons(net, percentage):
	counter = 1-percentage
	neurons = list(net.nodes)
	np.random.shuffle(neurons)
	inhibitory_neurons = []
	for idx, neuron in enumerate(neurons):
		counter = counter-percentage
		if(counter < 0):
			print('inhibitory neuron', idx)
			inhibitory_neurons.append(idx)
			counter = counter+1
			for edge in net.out_edges(neuron, data=True):
				synapse = edge[2]['object']
				synapse.w = -5*synapse.w
	return net,inhibitory_neurons
	
	
 
#net = initComplete()
net = initPartial()
#net = insertInhibition(net, 0.4)
net,inhibitory_neurons = insertInhibitionNeurons(net, 0.2)

spikes = []
voltage = []
calcium = []
reservoir = []
for neuron in net.nodes:
	spikes.append([])
	voltage.append([])
	calcium.append([])
	
axons = [SpikeDelay(np.random.randint(5,15), False) for _ in net.nodes]
constant_input = SpikeDelay(100, False)
constant_input.set(True)

training = 0000
steps = 300000

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
print('training done')

for t in range(steps):
	#pulse = 5 - (t/10000)
	#if(pulse > 0):
	#	list(net)[0].input(pulse)
	constant_input.set(False)
	if(constant_input.current()):
		list(net)[0].input(50)
		constant_input.set(True)
	
	total_synaptic_flux = 0
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
				
				synaptic_flux = synapse.effect(time+t)
				if(synaptic_flux > 0):
					synaptic_flux = synaptic_flux*global_reservoir
					total_synaptic_flux = total_synaptic_flux + synaptic_flux
				ca = edge[1].input(synaptic_flux)
				synapse.learn(ca)
				#print(synapse.w, ca)
				
		voltage[idxNeuron].append(neuron.v)
		calcium[idxNeuron].append(neuron.c)
	global_reservoir = global_reservoir + (1-global_reservoir)*0.001 - global_reservoir*total_synaptic_flux/1000
	if((t % 1000)==0):
		print(t, 'global', global_reservoir)
	reservoir.append(global_reservoir)
#print (spikes)
print('calc done')
#print('Cycles: ', len(list(nx.simple_cycles(net))))

#plt.figure(1)
#plt.title('Topology plot')
#nx.draw(net, with_labels = False)

#plt.figure(2)
#plt.eventplot(spikes, linelengths = [0.5 for n in net.nodes])     
#plt.title('Spike raster plot')
#plt.xlabel('Neuron')
#plt.ylabel('Spike')

fig, axs = plt.subplots(len(voltage)+1, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

plt.title('Membrane potential plot')
plt.xlabel('Neuron')
plt.ylabel('v')
for idx in range(neuron_count):
	#plt.eventplot(spikes, linelengths = [0.5 for n in net.nodes])    
	axs[idx].plot(np.arange(2*steps)/2, voltage[idx], 'xkcd:stone')
	axs[idx].plot(np.arange(steps), calcium[idx], 'xkcd:blue')
	if(idx in inhibitory_neurons):
		color = 'xkcd:red'
	else:
		color = 'xkcd:green'
	axs[idx].eventplot(spikes[idx], linelengths=[150], colors=[color])
		
	axs[idx].set_ylim(-100,100)

axs[neuron_count].plot(np.arange(steps), reservoir, 'xkcd:blue')
axs[neuron_count].set_ylim(-0.1,1.1)

plt.show()

