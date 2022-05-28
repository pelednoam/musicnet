def plot_midi_spectrogram(mels, title):
    specshow(mels, x_axis='time', y_axis='mel')
    plt.hlines([lbr.note_to_hz('C0'), lbr.note_to_hz('C1'), lbr.note_to_hz('C2'), lbr.note_to_hz('C3'), lbr.note_to_hz('C4'),
                lbr.note_to_hz('C5'), lbr.note_to_hz('C6'), lbr.note_to_hz('C7'), lbr.note_to_hz('C8')], 0, 10, linewidth=2)
    plt.xlim(0, 10)
    plt.title(title)
    plt.show()


def calc_powers_dists_mat(n_jobs=4):
    songs_fname = op.join(PROJECT_FOL, 'songs.pkl')
    distances_fname = op.join(PROJECT_FOL, 'distances.npy')
    songs = utils.load(songs_fname)
    couples = list(combinations(range(len(songs)), 2))
    indices = np.array_split(np.arange(len(couples)), n_jobs)
    chunks = [(indices_chunk, songs) for indices_chunk in indices]
    results = utils.run_parallel(_calc_powers_dists_mat, chunks, n_jobs)
    distances = np.zeros((len(songs), len(songs)))
    for distances_chunk in results:
        distances += distances_chunk
    np.save(distances_fname, distances)
    plt.matshow(distances)
    plt.savefig(op.join(PROJECT_FOL, 'songs_powers_cov_distances.jpg'))
    plt.close()


def _calc_powers_dists_mat(p):
    indices_chunk, songs = p
    distances = np.zeros((len(songs), len(songs)))
    for song1_ind, song2_ind in tqdm(indices_chunk):
        song1, song2 = songs[song1_ind], songs[song2_ind]
        song1_cov, song2_cov = np.cov(song1['powers']), np.cov(song2['powers'])
        distances[song1_ind, song2_ind] = np.linalg.norm(song1_cov - song2_cov)
    return distances


def calc_powers_dists_mat_old():
    songs_fname = op.join(PROJECT_FOL, 'songs.pkl')
    distances_fname = op.join(PROJECT_FOL, 'distances.npy')
    songs = utils.load(songs_fname)
    distances = np.zeros((len(songs), len(songs)))
    couples = list(combinations(range(len(songs)), 2))
    for song1_ind, song2_ind in tqdm(couples):
        song1, song2 = songs[song1_ind], songs[song2_ind]
        song1_cov, song2_cov = np.cov(song1['powers']), np.cov(song2['powers'])
        distances[song1_ind, song2_ind] = np.linalg.norm(song1_cov - song2_cov)
    # The distances matrix is symmetric
    distances += distances.T
    np.save(distances_fname, distances)
    plt.matshow(distances)
    plt.savefig(op.join(PROJECT_FOL, 'songs_powers_cov_distances.jpg'))
    plt.close()


def multi_dimensional_scaling(n_components=2, overwrite=False, n_jobs=4):
    from sklearn import manifold

    distances_fname = op.join(PROJECT_FOL, 'distances.npy')
    distances = np.load(distances_fname)

    pos_fname = op.join(PROJECT_FOL, 'multi_dimensional_scaling_pos.npy')
    if op.isfile(pos_fname) and not overwrite:
        pos = np.load(pos_fname)
        return pos

    mds = manifold.MDS(
        n_components=n_components, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=n_jobs)
    model = mds.fit(distances)
    pos = model.embedding_
    np.save(pos_fname, pos)


def plot_multi_dimensional_scaling():
    import matplotlib.patches as mpatches

    pos_fname = op.join(PROJECT_FOL, 'multi_dimensional_scaling_pos.npy')
    songs_fname = op.join(PROJECT_FOL, 'songs.pkl')
    pos = np.load(pos_fname)
    songs = utils.load(songs_fname)

    composers = list(set([song['composer'] for song in songs]))
    unique_colors = {
        composer:plt.cm.Spectral(col) for composer, col
        in zip(composers, np.linspace(0, 1, len(composers)))
    }
    colors = [unique_colors[song['composer']] for song in songs]
    ax = plt.figure(0, figsize=(12,8)).add_subplot(111)
    ax.scatter(pos[:, 0], pos[:, 1], c=colors)
    handles, labels = ax.get_legend_handles_labels()
    for composer, color in unique_colors.items():
        handles.append(mpatches.Patch(color=color, label=composer))
    ax.legend(handles=handles)
    plt.show()
    print('asfd')



# def read_midi_training_files_old():
#     # https://colab.research.google.com/github/BShakhovsky/PolyphonicPianoTranscription/blob/master/4%20Piano%20Audio%20to%20Midi.ipynb#scrollTo=zHqJ6IxDMoUq
#     songs = []
#     songs_fname = op.join(PROJECT_FOL, 'songs.pkl')
#     composers = [op.basename(d) for d in glob.glob(op.join(PROJECT_FOL, '*')) if op.isdir(d)]
#     midi_sampling_rate = None
#     for composer in composers:
#         for song_fname in glob.glob(op.join(PROJECT_FOL, composer, '*.wav')):
#             file_name = op.basename(song_fname)
#             print('Analyzing {}'.format(file_name))
#             if midi_sampling_rate is None:
#                 midi_sampling_rate = lbr.get_samplerate(song_fname)
#             song = lbr.load(song_fname, sr=midi_sampling_rate)[0]
#             mels = lbr.power_to_db(
#                 lbr.magphase(lbr.feature.melspectrogram(
#                     y=song, sr=midi_sampling_rate, n_mels=256, fmin=30, htk=True))[0])
#             songs.append(dict(
#                 fname=song_fname, powers=mels, composer=composer
#             ))
#             # plot_midi_spectrogram(mels, op.basename(song_fname))
#
#     print('Writing songs in {}'.format(songs_fname))
#     utils.save(songs, songs_fname)
