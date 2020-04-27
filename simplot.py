# Copyright (C) 2018 Florian Brandherm
# This file is part of flbrandh/MEC-Simulator <https://github.com/flbrandh/MEC-Simulator>.
#
# flbrandh/MEC-Simulator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# flbrandh/MEC-Simulator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with flbrandh/MEC-Simulator.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import simulator
import pygame
import pygame.gfxdraw
import math

class SimPlotStats:
    def __init__(self, sim):
        self.sim = sim
        self.fig, self.axGraph = plt.subplots(1,1)
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def plot(self):
        self.plotStatistics(self.sim, self.fig, self.axGraph)

    def chunksAvg(self, l, n):
        out = []
        for i in range(0, len(l), n):
            out.append(sum(l[i:i + n]) / len(l[i:i + n]))
        return out

    def chunksAvgZeroFiltered(self, l, n):
        out = []
        for i in range(0, len(l), n):
            filtered = [x for x in l[i:i + n] if not x==0]
            if len(filtered) == 0:
                out.append(0)
            else:
                out.append(sum(filtered) / len(filtered))
        return out

    def plotStatistics(self, sim, fig, ax):
        ax.clear()
        plt.sca(ax)

        stats = sim.getStatistics()
        avgInterval = 100
        #plt.plot(self.chunksAvg(stats.cost,avgInterval), 'r-', label='cost')
        #plt.plot(stats.inactiveRate, 'g.', label='inact. rate')
        #plt.plot(self.chunksAvg(stats.dissatisfactionRate,avgInterval),'b:', label='dissat. rate')
        plt.plot(self.chunksAvg([m / 20 for m in stats.num_proposed_migrations],avgInterval), 'yP', label='# proposed migrations/20')
        plt.plot(self.chunksAvg([m / 20 for m in stats.num_migrations],avgInterval), 'k+', label='# migrations/20')
        plt.plot(self.chunksAvgZeroFiltered(stats.avg_migration_dist_to_user_bs,avgInterval), 'm*', label='avg. dist. after mig.')
        plt.plot(self.chunksAvgZeroFiltered([m / 10 for m in stats.avg_latency], avgInterval), 'g-',label='avg. latency/10')

        #plt.plot([m/20 for m in stats.num_migration_events], 'g-', label='migration events/20')
        #plt.plot([a/b for a,b in zip(stats.num_proposed_migrations,stats.num_migration_events)], 'k-',  label='migration ratio')

        #ax.set_xlim(0, 1000)
        #ax.set_ylim(0, 1.1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol=5)
        ax.grid(color='y', linestyle=':')

        fig.canvas.draw()
        plt.pause(1e-7)

class SimPlotLearning(SimPlotStats):
    def __init__(self, sim, learner):
        self.sim = sim
        self.fig, [self.axSimStatsGraph, self.axLearnStatsGraph] = plt.subplots(2,1)
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()
        self.learner = learner

    def plot(self):
        self.plotLearning(self.learner, self.fig, self.axLearnStatsGraph)
        self.plotStatistics(self.sim, self.fig, self.axSimStatsGraph)
        self.fig.savefig("lastplot.pdf", bbox_inches='tight')


    def plotLearning(self, learner, fig, ax):
        ax.clear()
        plt.sca(ax)

        plt.plot(learner.avgRewards, 'b-',
                 label='average rewards')

        #if hasattr(learner, 'maxRewards'):
        #    plt.plot(learner.maxRewards, 'g-',
        #             label='maximum reward')
        #if hasattr(learner, 'minRewards'):
        #    plt.plot(learner.minRewards, 'r-',
        #             label='minimum reward')

        #ax.set_xlim(0, 1000)
        #ax.set_ylim(0, 1.1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=3)
        ax.grid(color='r', linestyle=':')

        #fig.canvas.draw()
        #plt.pause(1e-7)


class SimPlot(SimPlotStats):
    def __init__(self, sim):
        self.sim = sim
        self.fig, [self.axNetwork, self.axGraph] = plt.subplots(1, 2)
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def plot(self):
        self.plotSimulation(self.sim, self.fig, self.axNetwork)
        self.plotStatistics(self.sim, self.fig, self.axGraph)

    def plotSimulation(self, sim, fig, ax):
        clouds = sim.get_clouds()
        users = sim.get_users()
        nodes = sim.get_nodes()

        ax.clear()

        scale = 1
        ax.set_aspect('equal')
        #ax.autoscale(False)

        #plot connections between nodes
        for node in nodes:
            x1, y1 = node.get_pos()
            for neighbor in node.getNeighbors():
                x2, y2 = neighbor.get_pos()
                ax.plot([x1,x2],[y1,y2],'b')


        #plot connections to base stations
        for user in users:
            x1, y1 = user.get_pos()
            x2, y2 = user.get_base_station().get_pos()
            ax.plot([x1,x2],[y1,y2],'g')

        # plot internal nodes
        x = [node.get_pos()[0] for node in nodes if not isinstance(node, simulator.BaseStation)]
        y = [node.get_pos()[1] for node in nodes if not isinstance(node, simulator.BaseStation)]

        cloud_image_path = 'in.png'
        self.imscatter(x, y, cloud_image_path, zoom=0.2 * scale, ax=ax)

        # plot base stations
        x = [node.get_pos()[0] for node in nodes if
             isinstance(node, simulator.BaseStation)]
        y = [node.get_pos()[1] for node in nodes if
             isinstance(node, simulator.BaseStation)]

        cloud_image_path = 'bs.png'
        self.imscatter(x, y, cloud_image_path, zoom=0.2 * scale, ax=ax)

        # plot clouds
        x = [cloud.get_pos()[0] for cloud in clouds]
        y = [cloud.get_pos()[1] for cloud in clouds]

        cloud_image_path = 'ec.png'
        self.imscatter(x, y, cloud_image_path, zoom=0.2*scale, ax=ax)


        # plot users
        x = [user.get_pos()[0] for user in users]
        y = [user.get_pos()[1] for user in users]
        cloud_image_path = 'ue.png'
        self.imscatter(x, y, cloud_image_path, zoom=0.2*scale, ax=ax)

        #print cloud utilization
        for cloud in clouds:
            ax.text(cloud.get_pos()[0], cloud.get_pos()[1] + 0.05, '{0:.2f}'.format((cloud.totalMemoryRequirement() / cloud.memoryCapacity) * 100) + "%", color='k', bbox=dict(boxstyle="round",
                                                                                                                                                                               ec=(1., 0.5, 0.5),
                                                                                                                                                                               fc=(1., 0.8, 0.8),
                                                                                                                                                                               ))

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        fig.canvas.draw()
        plt.pause(1e-7)
        #plt.draw()
        #plt.show()

    def imscatter(self, x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        try:
            image = plt.imread(image)
        except TypeError:
            # Likely already an array...
            pass
        im = OffsetImage(image, zoom=zoom, interpolation='bicubic')
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists


class SimPlotPygame:

    def __init__(self, sim):
        self.sim = sim
        self.n = 0
        print("initializing Pygame thread")
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        self.drag=False
        self.mousePos = (0,0)
        self.quit = False

        self.imgEC = pygame.transform.scale(pygame.image.load('img/ec.png'),(64, 64))
        self.imgBS = pygame.transform.scale(pygame.image.load('img/bs.png'), (64, 64))
        self.imgIN = pygame.transform.scale(pygame.image.load('img/in.png'), (64, 64))
        self.imgUE = pygame.transform.scale(pygame.image.load('img/ue.png'), (64, 64))

        self.offset = (0.1,0.1)
        self.scale = 800

        pygame.font.init()
        self.subscriptFont = pygame.font.SysFont('FreeSans.ttf', 18)
        self.captionFont = pygame.font.SysFont('FreeSans.ttf', 40)

        self.display_histogram = False
        self.display_cloud_neighborhood = False
        self.display_cloud_services = False
        self.selected_cloud = 0
        self.display_service_cloud = True
        self.selected_service = 0

    def worldCoords2CamCoords(self,worldCoords,constantOffset=(0,0)):
        camCoords = (int(self.scale * (worldCoords[0] + self.offset[0]) +
                     constantOffset[0]),
                     int(self.scale * (worldCoords[1] + self.offset[1]) +
                     constantOffset[1]))
        return camCoords

    def drawImage(self, image, worldCoords, centered=True, constantOffset=(0,0)):

        camCoords = self.worldCoords2CamCoords(worldCoords,constantOffset)
        if centered:
            camCoords = (camCoords[0]-0.5*image.get_rect().size[0], camCoords[1]-0.5*image.get_rect().size[1])
        self.screen.blit(image, camCoords)

    def draw_world_space_circle(self, radius, color, worldCoords, constantOffset=(0,0)):
        cam_coords = self.worldCoords2CamCoords(worldCoords,constantOffset)
        radius = radius*self.scale
        pygame.gfxdraw.aacircle(self.screen, cam_coords[0],cam_coords[1], int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, cam_coords[0],
                                cam_coords[1], int(radius)-1, color)
        pygame.gfxdraw.aacircle(self.screen, cam_coords[0],
                                cam_coords[1], int(radius)-2, color)
        pygame.gfxdraw.aacircle(self.screen, cam_coords[0],
                                cam_coords[1], int(radius) - 3, color)

    def draw_world_space_line(self, worldCoordsStart, worldCoordsEnd, color, width=1):
        camCoordsStart = self.worldCoords2CamCoords(worldCoordsStart)
        camCoordsEnd = self.worldCoords2CamCoords(worldCoordsEnd)
        pygame.draw.lines(self.screen, color, False,
                          [camCoordsStart, camCoordsEnd], width)

    def drawTransparentBox(self, w, h, x, y, alpha, color):

        s = pygame.Surface((w, h))  # the size of your rect
        s.set_alpha(alpha)  # alpha level
        s.fill(color)  # this fills the entire surface
        self.screen.blit(s, (
        x,y))  # (0,0) are the top-left coordinates

    def drawHistogram(self, bins, labels, scale, caption, print_bins):

        axis_color = (0,0,0)
        background_color = (200,200,200)
        bar_color = (200,0,0)
        background_alpha = 230
        offsetY = 500
        offsetX = 30
        width = 800
        height = 400

        fontHeight = self.subscriptFont.get_height()

        self.drawTransparentBox(width+10,height+10+fontHeight,offsetX-10, offsetY-height, background_alpha, background_color)

        numBins = len(bins)
        for num, (bin, label) in enumerate(zip(bins,labels)):
            binW = width/numBins
            binH = height*(bin/scale)
            binXstart = (width/numBins) * num
            pygame.draw.rect(self.screen, bar_color,(offsetX+binXstart+1, offsetY-binH, binW-1, math.ceil(binH)))
            binXcenter = offsetX + binXstart + binW * 0.5
            if print_bins:
                text = self.subscriptFont.render(str(bin), True, axis_color)
                labelW, labelH = text.get_size()
                self.screen.blit(text, (
            binXcenter - labelW * 0.5, offsetY - 12))
            if label:
                pygame.draw.lines(self.screen, axis_color, False,
                                  [(binXcenter, offsetY + 10),
                                   (binXcenter, offsetY)], 2)
                text = self.subscriptFont.render(label, True, axis_color)
                labelW, labelH = text.get_size()
                self.screen.blit(text, (binXcenter-labelW*0.5, offsetY+10))

        pygame.draw.lines(self.screen, axis_color, False,
                          [(offsetX - 10, offsetY),
                           (offsetX + width, offsetY)], 2)
        pygame.draw.lines(self.screen, axis_color, False,
                          [(offsetX, offsetY + 10),
                           (offsetX, offsetY - height)], 2)

        caption = self.captionFont.render(caption,True,axis_color)
        captionW, captionH = caption.get_size()
        self.screen.blit(caption, (offsetX+width*0.5-captionW*0.5, offsetY-height))

        pygame.draw.rect(self.screen, axis_color, (offsetX-10, offsetY-height,width+10,height+10+fontHeight), True)

    def draw_cloud_neighborhood(self):
        clouds = self.sim.get_clouds()
        cloud = clouds[self.selected_cloud % len(clouds)]
        get_neigbor_clouds = getattr(cloud.get_migration_algorithm_instance(),"get_neighbor_clouds",None)
        if callable(get_neigbor_clouds):
            neighbor_clouds = cloud.get_migration_algorithm_instance().get_neighbor_clouds()
            for neighbor_cloud in neighbor_clouds:
                self.draw_world_space_circle(0.05, (255, 0, 0), neighbor_cloud.get_pos(), constantOffset=(32,32))

    def draw_cloud_services(self):
        clouds = self.sim.get_clouds()
        cloud = clouds[self.selected_cloud % len(clouds)]
        for service in cloud.get_services():
            pos = service.get_user().get_base_station().get_pos()
            self.draw_world_space_circle(0.05, (50, 100, 50),pos)



    def draw_cloud_annotations(self):
        #mark the seected cloud
        if self.display_cloud_neighborhood or self.display_cloud_services:
            clouds = self.sim.get_clouds()
            cloud = clouds[self.selected_cloud % len(clouds)]
            self.draw_world_space_circle(0.05, (200, 0, 155), cloud.get_pos(), constantOffset=(32, 32))

        if self.display_cloud_neighborhood:
            self.draw_cloud_neighborhood()
        if self.display_cloud_services:
            self.draw_cloud_services()

    def draw_service_annotations(self):
        if self.display_service_cloud:
            services = self.sim.get_services()
            service = services[self.selected_service%len(services)]
            self.draw_world_space_circle(.05, (255,0,0),service.get_user().get_pos())
            self.draw_world_space_circle(.05, (0, 255, 0),service.get_user().get_base_station().get_pos())
            self.draw_world_space_circle(.05, (255, 255, 0),service.get_cloud().get_pos(), constantOffset=(32, 32))

    def draw_multiline_text(self, str, font, x, y, color=(0,0,0)):
        strs = str.split('\n')
        offset_y = y
        for str in strs:
            text = font.render(str, True, color)
            self.screen.blit(text, (x, y+offset_y))
            offset_y += text.get_size()[1]


    def redraw(self):
        #update
        self.n += 10
        #self.scale += 10*math.sin(0.003*self.n)
        #self.offset = (self.offset[0] + 0.0005*math.sin(0.002*self.n),self.offset[1] + 0.0005*math.sin(0.0023*self.n))

        self.screen.fill((255,255,255))

        # draw grid
        for x in range(11):
            self.draw_world_space_line((0.1 * x, -1000), (0.1 * x, 1000), (230, 230, 255))
        for y in range(11):
            self.draw_world_space_line((-1000, 0.1 * y), (1000, 0.1 * y), (230, 230, 255))

        # plot connections between nodes
        for node in self.sim.get_nodes():
            start = node.get_pos()
            for neighbor in node.getNeighbors():
                end = neighbor.get_pos()
                self.draw_world_space_line(start, end, (0, 0, 200), 2)

        # plot connections to base stations
        for user in self.sim.get_users():
            start = user.get_pos()
            end = user.get_base_station().get_pos()
            self.draw_world_space_line(start, end, (200, 0, 0), 2)

        # plot internal nodes
        for node in self.sim.get_nodes():
            if not isinstance(node, simulator.BaseStation):
                self.drawImage(self.imgIN, node.get_pos())

        # plot base stations
        for node in self.sim.get_nodes():
            if isinstance(node, simulator.BaseStation):
                self.drawImage(self.imgBS, node.get_pos())

        # plot edge clouds
        for cloud in self.sim.get_clouds():
            self.drawImage(self.imgEC, cloud.get_pos(), centered=True, constantOffset=(32, 32))

        # plot users
        for user in self.sim.get_users():
            self.drawImage(self.imgUE, user.get_pos(), centered=True)

        self.draw_cloud_annotations()
        self.draw_service_annotations()

        # plot resource usage histogram
        if self.display_histogram:
            numBins = 10
            binSize = 0.2
            upperBound = numBins*binSize
            numOver200percentUtilized = 0
            resourceUsages = [0]*numBins
            resourceUsagesLabels = [ str(int(i*binSize*100))+'%-'+str(int((i+1)*binSize*100))+'%' for i in range(numBins)]
            for cloud in self.sim.get_clouds():
                resourceAvailable = cloud.memory_capacity()
                resourceUsed = cloud.totalMemoryRequirement()
                utilization = resourceUsed/resourceAvailable
                bin = int((numBins*(utilization/upperBound)))
                if bin >= numBins:
                    numOver200percentUtilized +=1
                else:
                    resourceUsages[bin] +=1
            resourceUsages.append(numOver200percentUtilized)
            resourceUsagesLabels.append('>200%')
            self.drawHistogram(resourceUsages,resourceUsagesLabels,25, "Cloud Memory Utlization", True)

        usageStr = \
        'display usage histogram: U\n' +\
        'toggle neighborhood: N\n' + \
        'toggle services: S\n' + \
        'change selected cloud: <,>\n' + \
        'toggle service placement: P\n' + \
        'change selected service: K,L\n' + \
        'selected cloud: ' + str(self.selected_cloud % len(self.sim.get_clouds())) + '\n' + \
        'selected service: ' + str(self.selected_service % len(self.sim.get_services()))

        self.draw_multiline_text(usageStr, self.subscriptFont,0,0,(0,0,0))

        pygame.display.update()

        return

    def plot(self):

        if pygame.event.get(pygame.QUIT):
            exit(0)

        for e in pygame.event.get():
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1: self.drag = True
                if e.button == 4: self.scale = min(100000, self.scale+100)
                if e.button == 5: self.scale = max(1, self.scale-100)

            if e.type == pygame.MOUSEBUTTONUP:
                self.drag = False

            if e.type == pygame.MOUSEMOTION:
                if self.drag:
                    deltaX, deltaY = e.pos
                    deltaX -= self.mousePos[0]
                    deltaY -= self.mousePos[1]
                    self.offset = (self.offset[0]+deltaX/self.scale, self.offset[1]+deltaY/self.scale)
                self.mousePos = e.pos

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_u:
                    self.display_histogram = not self.display_histogram

                elif e.key == pygame.K_n:
                    self.display_cloud_neighborhood = not self.display_cloud_neighborhood
                elif e.key == pygame.K_s:
                    self.display_cloud_services = not self.display_cloud_services
                elif e.key == pygame.K_COMMA:
                    self.selected_cloud -= 1
                elif e.key == pygame.K_PERIOD:
                    self.selected_cloud += 1

                elif e.key == pygame.K_p:
                    self.display_service_cloud = not self.display_service_cloud
                elif e.key == pygame.K_k:
                    self.selected_service -=1
                elif e.key == pygame.K_l:
                    self.selected_service +=1

        self.redraw()
        return


    def plotStatistics(self, sim, fig, ax):
        return

    def plotSimulation(self, sim, fig, ax):
        return