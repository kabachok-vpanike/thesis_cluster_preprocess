import json
import math
import sys
import numpy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import key_estimation
from sklearn.cluster import KMeans


def is_simple_chord(chord):
    degrees = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    scale_flat = [i + 'b' for i in degrees]
    scale_sharp = [i + '#' for i in degrees]
    all_notes = degrees + scale_flat + scale_sharp
    simple_chords = all_notes + [i + 'm' for i in all_notes]
    return chord in simple_chords


def note_to_int(note):
    note_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3,
                'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
                'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    return note_map[note]


def scale_degrees_to_roman(degree, alteration, is_minor, key_context):
    roman_numerals_major = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    roman_numerals_minor = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']

    if key_context == 'major':
        base_roman = roman_numerals_major[degree] if not is_minor else roman_numerals_major[degree].lower()
    else:
        if degree == 4 or degree == 6:
            base_roman = roman_numerals_minor[degree].upper()
        else:
            base_roman = roman_numerals_minor[degree] if is_minor else roman_numerals_minor[degree].upper()

    if alteration == 1:
        base_roman += '♯'
    elif alteration == -1:
        base_roman += '♭'

    return base_roman


def transpose_chord_to_key(chord, key):
    major_scale_degrees = [0, 2, 4, 5, 7, 9, 11]
    minor_scale_degrees = [0, 2, 3, 5, 7, 8, 10]

    key_root = note_to_int(key[:-1] if key.endswith('m') else key)
    chord_root = note_to_int(chord[:-1] if chord.endswith('m') else chord)
    chord_is_minor = chord.endswith('m')
    key_context = 'minor' if key.endswith('m') else 'major'
    interval = (chord_root - key_root) % 12
    scale_degrees = minor_scale_degrees if key_context == 'minor' else major_scale_degrees

    if interval in scale_degrees:
        degree_index = scale_degrees.index(interval)
        alteration = 0
    else:
        closest, alteration = min([(sd, interval - sd) for sd in scale_degrees], key=lambda x: abs(x[1]))
        degree_index = scale_degrees.index(closest)
        if alteration == 1:
            alteration = -1
            degree_index = (degree_index + 1) % 7
        alteration = -1 if alteration < 0 else (1 if alteration > 0 else 0)
    return scale_degrees_to_roman(degree_index, alteration, chord_is_minor, key_context)


def translate_chords_to_roman(chords, key):
    return [transpose_chord_to_key(chord, key) for chord in chords]


bigrams_by_occurrence = {}
trigrams_by_occurrence = {}


def count_trigram(roman_chords):
    for i in range(len(roman_chords) - 2):
        triplet = roman_chords[i:i + 3]
        if len(set(triplet)) == 3:
            if str(triplet) not in trigrams_by_occurrence:
                trigrams_by_occurrence[str(triplet)] = 0
            else:
                trigrams_by_occurrence[str(triplet)] += 1


def count_bigram(roman_chords):
    for i in range(len(roman_chords) - 1):
        duplet = roman_chords[i:i + 2]
        if str(duplet) not in bigrams_by_occurrence:
            bigrams_by_occurrence[str(duplet)] = 1
        else:
            bigrams_by_occurrence[str(duplet)] += 1


def matrix_distance(matrix1, matrix2, threshold):
    filtered_matrix1 = np.where(matrix1 > threshold, matrix1, 0)
    filtered_matrix2 = np.where(matrix2 > threshold, matrix2, 0)
    return np.linalg.norm(filtered_matrix1 - filtered_matrix2)


def weighted_distance(matrix1, matrix2, threshold):
    filtered_matrix1 = np.where(matrix1 > threshold, matrix1, 0)
    filtered_matrix2 = np.where(matrix2 > threshold, matrix2, 0)
    weights = 1 / (0.1 + np.abs(filtered_matrix1 - filtered_matrix2))
    weighted_diff = weights * (filtered_matrix1 - filtered_matrix2) ** 2
    distance = np.sum(weighted_diff)
    return distance


def get_representatives(centroid, transition_matrices, track_name_and_chord, songs_indices_global, cluster, n=5):
    top_representatives_data_with_thresholds = {}
    for threshold in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                      0.9, 0.95, 1.0]:
        distances = []
        for idx, matrix in enumerate(transition_matrices):
            distance = matrix_distance(matrix, centroid, threshold)
            distances.append((idx, distance))
        distances.sort(key=lambda x: x[1])
        top_n_representatives = [idx for idx, _ in distances[:n]]
        representatives_data = []
        for idx in top_n_representatives:
            representatives_data.append(track_name_and_chord[idx])

        all_representatives = [(idx, distance) for idx, distance in distances]

        for place, (idx, distance) in enumerate(all_representatives):
            if len(track_data_plus_cluster[songs_indices_global[idx]]) == 7:
                track_data_plus_cluster[songs_indices_global[idx]].append({cluster: [place]})
            else:
                if cluster not in track_data_plus_cluster[songs_indices_global[idx]][7]:
                    track_data_plus_cluster[songs_indices_global[idx]][7][cluster] = [place]
                else:
                    track_data_plus_cluster[songs_indices_global[idx]][7][cluster].append(place)
        top_representatives_data_with_thresholds[threshold] = representatives_data

    return top_representatives_data_with_thresholds


def process_exported_chords(exported_chords_file, possible_roman_chords):
    chords_num = len(possible_roman_chords)
    cnt_simple = 0
    chord_probability_all_songs = []
    with open(exported_chords_file) as file:
        chords_data = json.loads(file.read())
        print(len(chords_data))
        for track_data in chords_data:
            track_chords = limit_consecutive_elements(track_data[3].split(' '))
            if all(is_simple_chord(chord) for chord in track_chords):
                if cnt_simple % 10000 == 0:
                    print(cnt_simple)
                key = key_estimation.estimate_key(track_chords)
                roman_chords = translate_chords_to_roman(track_chords, key)
                cnt_simple += 1
                count_bigram(roman_chords)
                bigrams_by_occurrence_copy = bigrams_by_occurrence.copy()
                bigrams_by_occurrence.clear()

                chord_probability = numpy.array([chords_num * [0] for i in range(chords_num)], dtype=float)

                for i in range(chords_num):
                    for j in range(chords_num):
                        chord1 = possible_roman_chords[i]
                        chord2 = possible_roman_chords[j]
                        chord_probability[i][j] = bigrams_by_occurrence_copy[str([chord1, chord2])] if str(
                            [chord1, chord2]) in bigrams_by_occurrence_copy else 0
                        chord_probability[j][i] = bigrams_by_occurrence_copy[str([chord2, chord1])] if str(
                            [chord2, chord1]) in bigrams_by_occurrence_copy else 0

                chord_probability_to_save = {}

                for i in range(chords_num):
                    row_sum = sum(chord_probability[i, :])
                    if row_sum > 0:
                        chord_probability[i, :] = chord_probability[i, :].astype(float) / row_sum
                        for j in range(chords_num):
                            chord_probability_to_save[f"{i} {j}"] = chord_probability[i][j]

                chord_probability_all_songs.append(
                    (chord_probability_to_save, ' '.join(track_chords),
                     track_data[4], track_data[5], roman_chords, track_data[0]))

    return chord_probability_all_songs


def save_chord_probability_all_songs(filename, chord_probability_all_songs):
    with open(filename, 'w') as f:
        json.dump(chord_probability_all_songs, f)


track_data_plus_cluster = []


def get_chord_probability_all_songs(filename):
    chord_probability_all_songs = []
    track_name_and_chord = []
    with open(filename, 'r') as f:
        chord_probability_all_songs_from_save = json.load(f)
        for chord_map in chord_probability_all_songs_from_save[:100]:
            song_chord_probability = np.zeros((chords_num, chords_num))
            track_name_and_chord.append((chord_map[1], chord_map[2], chord_map[3], chord_map[4], chord_map[5]))
            track_data_plus_cluster.append(chord_map)
            for entry in chord_map[0]:
                chord1 = int(entry.split(' ')[0])
                chord2 = int(entry.split(' ')[1])
                song_chord_probability[chord1][chord2] = chord_map[0][entry]
            chord_probability_all_songs.append(song_chord_probability)
    return np.array(chord_probability_all_songs), track_name_and_chord


def get_chord_probability_for_cluster(chord_probability_all_songs, cluster, track_name_and_chord):
    chord_probability = []
    songs_indices_global = []
    track_info = []
    count_songs_in_cluster = 0
    for i in range(len(chord_probability_all_songs)):
        if cluster == cluster_labels[i]:
            count_songs_in_cluster += 1
            if len(track_data_plus_cluster[i]) == 6:
                track_data_plus_cluster[i].append([cluster])
            else:
                track_data_plus_cluster[i][6].append(cluster)
            chord_probability.append(chord_probability_all_songs[i])
            track_info.append(track_name_and_chord[i])
            songs_indices_global.append(i)
    return chord_probability, track_info, count_songs_in_cluster, songs_indices_global


def add_cluster_to_canvas(chord_probability, possible_roman_chords, axs, clusters_graphs):
    chords_num = len(possible_roman_chords)
    cluster_graph = {}
    G = nx.DiGraph()
    for i in range(chords_num):
        for j in range(i, chords_num):
            chord1 = possible_roman_chords[i]
            chord2 = possible_roman_chords[j]
            if chord_probability[i, j] > 0.05:
                if chord1 not in cluster_graph:
                    cluster_graph[chord1] = []
                cluster_graph[chord1].append([chord2, chord_probability[i, j]])
                G.add_edge(chord1, chord2, weight=3 * chord_probability[i, j])

            if chord_probability[j, i] > 0.05:
                if chord2 not in cluster_graph:
                    cluster_graph[chord2] = []
                cluster_graph[chord2].append([chord1, chord_probability[j, i]])
                G.add_edge(chord2, chord1, weight=3 * chord_probability[j, i])
    clusters_graphs.append(cluster_graph)
    pos = nx.spring_layout(G, k=1)
    nx.draw_networkx_nodes(G, pos, ax=axs[cluster], node_size=400)
    nx.draw_networkx_labels(G, pos, ax=axs[cluster], font_size=10)

    weights = nx.get_edge_attributes(G, 'weight')
    for (node1, node2, data) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(node1, node2)], ax=axs[cluster],
                               width=data['weight'], arrowstyle='->', arrowsize=10, connectionstyle="arc3,rad=0.4",
                               label=data['weight'])

    edge_labels = {(u, v): round(d['weight'] / 10, 2) for u, v, d in G.edges(data=True)}

    axs[cluster].set_title(f'Cluster {cluster + 1}')
    axs[cluster].axis('off')

    return axs, clusters_graphs


def print_plot(num_clusters, axs, plt, fig):
    for i in range(num_clusters, rows * cols):
        fig.delaxes(axs[i])
    plt.tight_layout()
    plt.show()


def limit_consecutive_elements(input_list, limit=2):
    if not input_list:
        return []

    new_list = [input_list[0]]

    for element in input_list[1:]:
        if new_list[-limit:] == [element] * limit:
            continue
        else:
            new_list.append(element)
    return new_list


if __name__ == '__main__':
    possible_roman_chords = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    possible_roman_chords += [i.lower() for i in possible_roman_chords]
    possible_roman_chords += [i + '♯' for i in possible_roman_chords] + [i + '♭' for i in possible_roman_chords]

    numpy.set_printoptions(threshold=sys.maxsize)

    chords_num = len(possible_roman_chords)

    # chord_probability_all_songs = process_exported_chords(exported_chords_file="exported-chords.json",
    #                                                      possible_roman_chords=possible_roman_chords)

    filenameToSave = 'data.json'

    # save_chord_probability_all_songs(filename=filenameToSave, chord_probability_all_songs=chord_probability_all_songs)

    chord_probability_all_songs, track_name_and_chord = get_chord_probability_all_songs(filename=filenameToSave)
    clusters_to_save = {}

    for num_clusters in range(1, 11):
        print(num_clusters)
        clusters_graphs = []
        count_songs_in_cluster = {}

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict([matrix.flatten() for matrix in chord_probability_all_songs])
        cols = int(math.ceil(math.sqrt(num_clusters)))
        rows = int(math.ceil(num_clusters / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if num_clusters > 1:
            axs = axs.flatten()
        else:
            axs = [axs]

        clusters_representatives_info = []
        for cluster in range(num_clusters):
            print("cluster: ", cluster)
            chord_probability, tracks_info_for_cluster, count_songs, songs_indices_global = get_chord_probability_for_cluster(
                chord_probability_all_songs,
                cluster,
                track_name_and_chord)
            count_songs_in_cluster[cluster] = count_songs
            chord_probabilities_in_cluster = chord_probability.copy()

            chord_probability = np.mean(np.array(chord_probability), axis=0)

            representatives_data = get_representatives(chord_probability,
                                                       chord_probabilities_in_cluster,
                                                       tracks_info_for_cluster,
                                                       songs_indices_global,
                                                       num_clusters,
                                                       5)
            clusters_representatives_info.append(representatives_data)
            axs, clusters_graphs = add_cluster_to_canvas(chord_probability, possible_roman_chords, axs, clusters_graphs)
        clusters_to_save[num_clusters] = [clusters_graphs, clusters_representatives_info, count_songs_in_cluster]
    filename = "clustersGraphs.json"
    with open(filename, 'w') as f:
        json.dump(clusters_to_save, f)

    # chord_probability_all_songs.tolist()
    # array_for_saving = []
    # for i in range(len(chord_probability_all_songs)):
    #     array_for_saving.append([chord_probability_all_songs[i].tolist(), *track_name_and_chord[i]])

    fileForData = 'data2.json'
    save_chord_probability_all_songs(filename=fileForData, chord_probability_all_songs=track_data_plus_cluster)

    only_raw_chords_info = [song_data[1] for song_data in track_data_plus_cluster]
