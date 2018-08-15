from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.stats import norm
from matplotlib import animation


class Person:
    def __init__(self, n_steps=200, p_vaccinated=1, p_healthy=1, max_x=10, max_y=10, D=.1):
        self.vaccinated = np.random.random() <= p_vaccinated
        self.healthy = [np.random.random() <= p_healthy]

        self.i = 0  # Simulated step for person
        self.n_steps = n_steps
        self.D = D
        self.max_x = max_x
        self.max_y = max_y
        self.locations = np.zeros((n_steps, 2))

    def calculate_all_locations(self):
        r_x = norm.rvs(size=(self.n_steps), scale=self.D)
        out_x = np.cumsum(r_x, axis=-1) + np.random.random()*self.max_x
        out_x = out_x % self.max_x
        r_y = norm.rvs(size=(self.n_steps), scale=self.D)
        out_y = np.cumsum(r_y, axis=-1) + np.random.random()*self.max_y
        out_y = out_y%self.max_y

        self.locations = np.vstack([out_x, out_y])


class Society:
    def __init__(self, n_persons, p_vaccinated, p_healthy, n_steps=200, D=.1):
        self.people = [Person(n_steps=n_steps,p_vaccinated=p_vaccinated, p_healthy=p_healthy, D=D) for _ in range(n_persons)]
        for p in self.people:
            p.calculate_all_locations()
        self.n_steps = n_steps

    def update_healthyness(self):
        arr = np.zeros((len(self.people), 2))
        for j in range(self.n_steps):
            for i in range(len(self.people)):
                arr[i][:] = self.people[i].locations[:,j]
            tree = spatial.KDTree(arr)
            nn = tree.query(arr, k=10, distance_upper_bound=1)
            distances = nn[0]
            neighbours = nn[1]
            for n_person in range(distances.shape[0]):
                dist = distances[n_person][:]
                indexes = np.argwhere(dist<.5)
                curr_person = self.people[n_person]
                curr_person.healthy.append(curr_person.healthy[-1])
                if curr_person.healthy[-1]:
                    for index in indexes:
                        person_neighbour = self.people[neighbours[n_person, index][0]]
                        if not person_neighbour.healthy[-1]:
                            curr_person.healthy[-1] = False
                            break


    def plot_people(self, frame):
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        for p in self.people:
            try:
                if p.healthy[frame]:
                    plt.plot(p.locations[0][frame], p.locations[1][frame], 'go')
                else:
                    plt.plot(p.locations[0][frame], p.locations[1][frame], 'ro')
            except:
                plt.plot(p.locations[0][frame], p.locations[1][frame], 'go')
        plt.show()

    def plot_person(self, p_n):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        p = self.people[p_n]
        plt.plot(p.locations[0][:], p.locations[1][:], 'go-')
        plt.show()


if __name__ == '__main__':

    society = Society(n_persons=100, p_vaccinated=1, p_healthy=1, n_steps=500, D=.2)
    society.people[0].healthy[-1] = False
    society.update_healthyness()
    for i in range(len(society.people[0].healthy)-1):
        fig, ax = plt.subplots()
        arr = np.zeros((len(society.people),2))
        c = np.zeros((len(society.people)))
        j = 0
        for p in society.people:
            arr[j,:] = p.locations[:,i]
            if p.healthy[i]:
                c[j] = 0
            else:
                c[j] = 1
            j+=1
        scat = plt.scatter(arr[:,0], arr[:,1], c=c)
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        filename = 'pngs_03\\Population_{:05d}.png'.format(i)
        plt.savefig(filename, dpi=48)
        plt.close()