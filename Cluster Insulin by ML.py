import pandas as pd
import numpy as np
from glob import glob
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
import os
import seaborn as sns

def rename_tinagrast():
    path = r'E:\My Codes\Clustering Insulin Masoud\Data\dpt\tinagrast'
    files = os.listdir(path)
    files.reverse()
    for file in files:
        mo = re.search(r'(\d+.*)(\d)(.*)', file)
        one, two, three = mo.groups()
        new_name = f'{one}{int(two)+1}{three}'
        print(file, new_name)
        os.rename(file, new_name)
        

def load_data(path):
    files = glob(path)
    absorbs = []
    waves = []
    info = []
    labels = []
    temps = []
    weeks = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()
            line = lines[0]
            if line.find('\t') != -1:
                sep = '\t'
            else:
                sep = ' '
        ir = pd.read_csv(file, sep=sep, names=['wavelength', 'abs'])
        abs_list = list(ir['abs'])
        wave_length = list(ir['wavelength'])

        # wave_length = wave_length[209:911]
        # abs_list = abs_list[209:911]
        
        wave_length = wave_length[559:2239]
        abs_list = abs_list[559:2239]
        
        txt = file.split('\\')[-1]
        pattern = r'(\d+)([a-z]+)(\d+).*.dpt'
        mo = re.search(pattern, txt)
        tempreture, drug, week = mo.groups()

        info.append((tempreture, str(week), drug))
        # if tempreture not in ['25', '0']:
        #     continue
        absorbs.append(abs_list)
        waves.append(wave_length)
        labels.append(txt.split('.dpt')[0].replace('dpt', ''))
        temps.append(tempreture)
        weeks.append(week)

    absorbs = np.array(absorbs)
    waves = np.array(waves)
    info = np.array(info)
    return absorbs, waves, info, labels, temps, weeks


def normalizing(waves, absorbs):
    # Before normalize

    # create an iterator of colors
    colors = iter(cm.rainbow(np.linspace(0, 1, len(absorbs))))
    for i, (wave, absorb) in enumerate(zip(waves, absorbs)):
        plt.plot(wave, absorb, c=next(colors))
    plt.title('before normalization')
    plt.show()

    # After SNVnormalize
    # create an iterator of colors
    colors = iter(cm.rainbow(np.linspace(0, 1, len(absorbs))))
    normalized = []
    for i, absorb in enumerate(absorbs):
        normed = (absorb - absorb.mean()) / absorb.std()
        normalized.append(normed)

        plt.plot(wave, normed, c=next(colors))
        # plt.show()
    plt.title('After SNV')
    plt.show()
    
    return normalized

def each_plot_save(waves, normalized, info_list):
    # to save all noramalized plots
    for normed, info, wave in zip(normalized, info_list, waves):
        txt = info[2] + info[0] + 'w' + info[1]
        
        fig = plt.figure(dpi=300)
        plt.plot(wave, normed, 'r')
        plt.title(f'{txt} normalized')
        fig.savefig(f'Result\\New folder\\{txt}.jpg')

def all_plot_in_one_plot(all_info):
    sns.set_theme(context='paper', font_scale=1.2, style='dark', palette='viridis')
    fig, axs = plt.subplots(nrows=16 , ncols=3, figsize=(15, 40), dpi=300)
    colors = sns.color_palette('viridis')
    for number, line in enumerate(all_info):
        absorbs, waves, info, drug_names, temps, weeks, normalized, branch_orders = line
        for i in range(len(info)):
            if i ==0:
                axs[i, number].set_title(info[0][2].title(), pad=10)
            if number==0:
                axs[i, number].set_ylabel(f'{temps[i]} week {weeks[i]}', rotation=0, labelpad=30)
            # print(drug_names[i], i)
            sns.lineplot(x=waves[i], y=normalized[i], color=colors[number], ax=axs[i, number])
    fig.savefig(f'Result\\all_plot_in_one.jpg')
    
def search_in_info(info, branch):
    temp, drug_name, week = re.search(r'(\d+)(.*?)(\d).*', branch).groups()
    for idx, line in enumerate(info):
        if line[0]==temp and line[1]==week:
            return idx


def all_plot_in_one_plot_in_branch_order(all_info):
    sns.set_theme(context='paper', font_scale=1.2, style='dark', palette='viridis')
    fig, axs = plt.subplots(nrows=16 , ncols=3, figsize=(26, 40), dpi=300)
    colors = sns.color_palette('viridis')
    for number, line in enumerate(all_info):
        absorbs, waves, info, drug_names, temps, weeks, normalized, branch_orders = line
        for i, branch in enumerate(branch_orders):
            idx = search_in_info(info, branch)
            if i ==0:
                axs[i, number].set_title(info[0][2].title(), pad=8)
            
            print(idx, drug_names[idx])
            axs[i, number].set_ylabel(f'{temps[idx]} week {weeks[idx]}', rotation=0, labelpad=30)
            sns.lineplot(x=waves[idx], y=normalized[idx], color=colors[number], ax=axs[i, number])
            
    fig.savefig(f'Result\\all_plot_in_one_in_branch_order.jpg')
    
def plot_just_temp(desired_temp, normalized, waves, temps, labels):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(normalized))))
    for idx, normed in enumerate(normalized):
        temp = temps[idx]
        if temp == desired_temp:
            plt.plot(waves[idx], normed, c=next(colors))
            plt.title(labels[idx])
            plt.show()

# %% Without PCA


def cluster_by_scipy(normalized, drug_names, n_clusters=3,
                     show_chart=False):
    # Calculate Euclidean distance
    dist_matrix = pdist(normalized, metric='euclidean')

    # Perform hierarchical clustering using Ward's method
    hc1 = linkage(dist_matrix, method='ward')

    if show_chart:
        # Plot the dendrogram
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size
        dendrogram(hc1, ax=ax, labels=drug_names,
                   leaf_rotation=0, leaf_font_size=8, orientation='left')  # Adjust rotation and font size

        ax.set_xlabel('Samples')
        ax.set_ylabel('Distance')
        ax.set_title('Thermal Similarity Scipy Dendrogram')

        plt.tight_layout()
        plt.show()

    # Determine the cluster labels for each data point
    pred = fcluster(hc1, n_clusters, criterion='maxclust')

    # Calculate the silhouette score
    sil_score = silhouette_score(normalized, pred, metric='euclidean')

    return sil_score


def cluster_by_kmeans(normalized, n_clusters, drug_names=None):

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=0, n_init='auto')

    pred = kmeans.fit_predict(normalized)
    centers = kmeans.cluster_centers_
    silhouette = silhouette_score(normalized, pred)
    ssd = kmeans.inertia_
    
    if drug_names:
        result_kmeans = np.array(list(zip(drug_names, pred)))
        plt.scatter(result_kmeans[:, 1].astype(np.int32), result_kmeans[:, 0],
                    c=[pred])
        plt.title('clusters kmean')
    
        plt.show()
    
    return silhouette, ssd, centers, pred

# %% Elbow charts
def elbow_chart_kmeans(upper_limit, normalized):

    ssds = []
    scores = []
    numbers = range(2, upper_limit)
    for n_clusters in numbers:
        print(f'\\n\n{n_clusters=}\n\n')
        silhouette, ssd, centers, pred = cluster_by_kmeans(
            normalized, n_clusters)
        ssds.append(ssd)
        scores.append(silhouette)

    plt.plot(numbers, scores, c='b')
    plt.title('silluette kmeans')
    plt.ylabel('score')
    plt.xlabel('n_clusters')
    plt.show()

    plt.plot(numbers, ssds, c='r')
    plt.title('ssd kmeans')
    plt.ylabel('ssd')
    plt.xlabel('n_clusters')
    plt.show()


def elbow_chart_hca(upper_limit, normalized, labels):

    scores = []
    numbers = range(2, upper_limit)
    for n_clusters in numbers:
        print(f'\\n\n{n_clusters=}\n\n')
        silhouette = cluster_by_scipy(normalized, labels,
                                      n_clusters, show_chart=False)
        scores.append(silhouette)

    plt.plot(numbers, scores, c='g')
    plt.title('silluette HCA')
    plt.ylabel('score')
    plt.xlabel('n_clusters')
    plt.show()


#%% With PCA
def cluster_by_pca_kmeans_sklearn(data, info, drug_names, n_clusters=4):

    # Perform PCA and reduce the dimensionality
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pca_data)
    labels = kmeans.labels_

    # Calculate the silhouette score
    score = silhouette_score(pca_data, labels)
    print(f'Silhouette score: {score:.2f}')

    # Visualize the clustering result
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(
        'PCA + K-means  Concentric Circles Kmeans Sklearn\nSilhouette Score: {:.2f}'.format(score))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Annotate the data labels
    # for i, label in enumerate(labels):
    for i, label in enumerate(info[:, :2]):
        label=' w'.join(label)
        plt.annotate(label, (pca_data[i, 0]+0.4, pca_data[i, 1]), fontsize=5)

    # Plot the centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)

    # Plot concentric circles
    origin = np.array([0, 0])
    max_distance = np.max(np.linalg.norm(pca_data - origin, axis=1))
    num_circles = 5
    circle_distances = np.linspace(0, max_distance, num_circles)

    for distance in circle_distances:
        circle = plt.Circle(origin, distance, color='blue',
                            fill=False, linestyle='dashed')
        plt.gca().add_patch(circle)

    plt.axis('equal')  # Ensure that the x and y axes have the same scale
    plt.show()




def cluster_by_pca_hca_sklearn(data, info, waves, drug_names, n_clusters=4):
    # Perform PCA and reduce the dimensionality
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Apply Agglomerative Clustering

    hca = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hca.fit_predict(pca_data)

    # Calculate the silhouette score
    score = silhouette_score(pca_data, labels)
    print(f'Silhouette score: {score:.2f}')

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(40, 16)
                                   )

    # Visualize the clustering result in the first subplot
    ax1.scatter(pca_data[:, 0], pca_data[:, 1], c='purple', s=50)
    ax1.set_title(
        'PCA + HCA clustering')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')

    # Annotate the data labels
    # for i, label in enumerate(labels):
    for i, label in enumerate(info[:, :2]):
        label=' w'.join(label)
        x, y = pca_data[i, 0], pca_data[i, 1]
        coords = (x, y + 0.003)
        ax1.annotate(label, coords, fontsize=8, color='black')
    
    # Plot concentric circles in the first subplot
    origin = np.array([0, 0])
    max_distance = np.max(np.linalg.norm(pca_data - origin, axis=1))
    num_circles = 5
    circle_distances = np.linspace(0, max_distance, num_circles)

    for distance in circle_distances:
        circle = plt.Circle(origin, distance, color='blue',
                            fill=False, linestyle='dashed')
        ax1.add_patch(circle)

    ax1.axis('equal')  # Ensure that the x and y axes have the same scale

    # Calculate the linkage matrix
    Z = linkage(pca_data, method='ward')

    # Plot the dendrogram in the second subplot
    result = dendrogram(Z, ax=ax2,labels=drug_names, orientation='left')

    
    ax2.set_title('Dendrogram of HCA')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Distance')

    plt.show()
    
    branch_orders = result['ivl'].copy()
    branch_orders.reverse()
    return branch_orders

def cluster_by_pca_hca_scipy(data, info, drug_names, n_clusters=3):

    # Perform PCA and reduce the dimensionality
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Apply hierarchical clustering using the 'ward' linkage method
    linked = linkage(pca_data, method='ward')

    # Calculate the cophenetic correlation coefficient to measure clustering quality
    c, coph_dists = cophenet(linked, pdist(pca_data))
    print(f'Cophenetic correlation coefficient: {c:.2f}')

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    plt.title(
        'PCA + HCA Dendrogram Scipy (Cophenetic Correlation Coefficient: {:.2f}\n)'.format(c))
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    result =dendrogram(linked, labels=drug_names, leaf_rotation=0, leaf_font_size=8,
               color_threshold=12, orientation='left')
    plt.show()

    # Get the labels using the fcluster function with a specific distance threshold
    distance_threshold = 12
    # labels = fcluster(linked, distance_threshold, criterion='distance')
    labels = fcluster(linked, n_clusters, criterion='maxclust')

    # Annotate the cluster labels
    # for i, label in enumerate(labels):
    #     plt.annotate(label, (pca_data[i, 0]+0.4, pca_data[i, 1]), fontsize=5)

    # Visualize the clustering result with concentric circles
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title('PCA + HCA scipy with Concentric Circles with Scipy')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Plot concentric circles
    origin = np.array([0, 0])
    max_distance = np.max(np.linalg.norm(pca_data - origin, axis=1))
    num_circles = 5
    circle_distances = np.linspace(0, max_distance, num_circles)

    for distance in circle_distances:
        circle = plt.Circle(origin, distance, color='blue',
                            fill=False, linestyle='dashed')
        plt.gca().add_patch(circle)

    plt.axis('equal')  # Ensure that the x and y axes have the same scale
    
    # for i, label in enumerate(labels):
    for i, label in enumerate(info[:, :2]):
        label=' w'.join(label)
        plt.annotate(label, (pca_data[i, 0]+0.4, pca_data[i, 1]), fontsize=5)

    plt.show()


def pca_kmean_not_center_circles(data, info, drug_names, n_clusters=3):
    # Perform PCA and reduce the dimensionality
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pca_data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Visualize the clustering result
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title('PCA + K-means Clustering with Cluster-Centered Circles')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Plot circles centered at cluster centroids
    for centroid in centroids:
        cluster_points = pca_data[labels ==
                                  np.where(centroids == centroid)[0][0]]
        max_distance = np.max(np.linalg.norm(
            cluster_points - centroid, axis=1))
        circle = plt.Circle(centroid, max_distance,
                            color='blue', fill=False, linestyle='dashed')
        plt.gca().add_patch(circle)
    
    plt.axis('equal')  # Ensure that the x and y axes have the same scale
    
    for i, label in enumerate(info[:, :2]):
        label=' w'.join(label)
        plt.annotate(label, (pca_data[i, 0]+0.4, pca_data[i, 1]), fontsize=5)

    plt.show()


def title_text_image(drug):
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.text(0.5, 0.5, drug.replace('dpt ', '').title(),
            horizontalalignment='center', verticalalignment='center',
            fontsize=20)
    plt.show()

# %% Main

def main(path, n_clusters):
    absorbs, waves, info, drug_names, temps, weeks = load_data(path)
    normalized = normalizing(waves, absorbs)
    
    # normalized = absorbs
    # each_plot_save(waves, normalized, info)
    # each_plot_save(waves, absorbs, info)
    cluster_by_scipy(normalized, drug_names, n_clusters, show_chart=True)

    # # plot_just_temp('25', normalized, waves, temps, labels)
    
    # UPPER_LIMIT = 15
    # elbow_chart_hca(UPPER_LIMIT, normalized, drug_names)
    # elbow_chart_kmeans(UPPER_LIMIT, normalized)
    
    # # i found 6 clusters is the best for kmeans by elbow chart
    cluster_by_kmeans(normalized, n_clusters, drug_names)
    
    # cluster_by_pca_kmeans_sklearn(normalized)
    cluster_by_pca_hca_scipy(normalized, info, drug_names, n_clusters)
    branch_orders = cluster_by_pca_hca_sklearn(normalized, info, waves, drug_names, n_clusters)
    pca_kmean_not_center_circles(normalized, info, drug_names, n_clusters)
    return absorbs, waves, info, drug_names, temps, weeks, normalized, branch_orders



drugs = os.listdir(r'Data\dpt')
all_info = []
for drug in drugs:
    title_text_image(drug)
    path = f'Data\\dpt\\{drug}\\*.dpt'
    result = main(path, n_clusters=2)
    all_info.append(result)
all_plot_in_one_plot(all_info)
all_plot_in_one_plot_in_branch_order(all_info)
