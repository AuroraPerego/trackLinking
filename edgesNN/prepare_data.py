#!/usr/bin/env python3
"""
Prepare Track-Trackster & Trackster-Trackster edge datasets from ROOT files.
Usage: python prepare_data.py --input /path/to/root/file.root --output data.npz
"""

import argparse
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.stats import mode

C_CM_PER_NS = 29.9792458

def decide_pid(pids):
    if 4 in pids:
        return 4
    if 5 in pids:
        return 5
    return mode(pids).mode

def find_track_id(array, number):
    try:
        return np.where(array == number)[0][0]
    except:
        return -1

def distance(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5

def compute_track_ts_edge_features(refPt, refP, refEta, refPhi, trk_time, trk_timeErr,
                                  tsEta, tsPhi, tsEnergy, tsTime, tsTimeErr, deltaTime):
    return np.column_stack([
        np.full_like(tsEta, refPt),
        np.full_like(tsEta, refP),
        np.full_like(tsEta, refEta),
        np.full_like(tsEta, np.sin(refPhi)),
        np.full_like(tsEta, np.cos(refPhi)),
        np.full_like(tsEta, trk_time),
        np.full_like(tsEta, trk_timeErr),
        tsEnergy, tsEta, np.sin(tsPhi), np.cos(tsPhi), tsTime, tsTimeErr, deltaTime
    ])

def compute_ts_ts_edge_features(eta1, phi1, E1, z1, time1, timeErr1, eta2, phi2, E2, z2, time2, timeErr2, deltaTime, samePid):
    return np.column_stack([
        E1, eta1, np.sin(phi1), np.cos(phi1), z1, time1, timeErr1,
        E2, eta2, np.sin(phi2), np.cos(phi2), z2, time2, timeErr2, deltaTime, samePid
    ])

def load_branch_with_highest_cycle(file, branch_name):
    all_keys = file.keys()
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
    return file[highest_cycle_key]

def main(args):
    files = [args.input] if isinstance(args.input, str) else args.input
    all_trk_features = []
    all_trk_labels = []
    all_ts_features = []
    all_ts_labels = []

    # Keys
    tracksKeys = ['track_id', 'track_hgcal_x', 'track_hgcal_y', 'track_hgcal_z', 'track_hgcal_eta', 'track_hgcal_phi', 'track_hgcal_pt', 'track_pt', 'track_p', 'track_missing_outer_hits', 'track_missing_inner_hits', 'track_quality', 'track_time_mtd', 'track_time_mtd_err', 'track_pos_mtd/track_pos_mtd.theVector.theX', 'track_pos_mtd/track_pos_mtd.theVector.theY', 'track_pos_mtd/track_pos_mtd.theVector.theZ', 'track_nhits', 'track_isMuon', 'track_isTrackerMuon']

    simTsKeys = ['regressed_energy', 'raw_energy', 'trackIdx', 'barycenter_z', 'barycenter_eta', 'barycenter_phi', 'CPidx', 'pdgID', 'vertices_indexes', 'vertices_x', 'vertices_y', 'vertices_z', 'vertices_time', 'vertices_energy', 'vertices_multiplicity']

    assKeys = ['ticlTracksterLinks_simToReco_CP', 'ticlTracksterLinks_simToReco_CP_score', 'ticlTracksterLinks_simToReco_CP_sharedE', 'ticlTracksterLinks_simToReco_SC', 'ticlTracksterLinks_simToReco_SC_score', 'ticlTracksterLinks_simToReco_SC_sharedE', 'ticlCandidate_simToReco_SC', 'ticlCandidate_simToReco_SC_score', 'ticlCandidate_simToReco_SC_sharedE', 'ticlCandidate_simToReco_CP', 'ticlCandidate_simToReco_CP_score', 'ticlCandidate_simToReco_CP_sharedE']

    tsKeys = ['time', 'timeError', 'regressed_energy', 'raw_energy', 'raw_em_energy', 'raw_pt', 'raw_em_pt', 'barycenter_x', 'barycenter_y', 'barycenter_z', 'barycenter_eta', 'barycenter_phi', 'id_probabilities', 'vertices_x', 'vertices_y', 'vertices_z', 'vertices_energy']


    for file_path in files:
        print(f"Loading {file_path}")
        file = uproot.open(file_path)

        # Load all branches
        alltracksters = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh')
        allsimtrackstersCP = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
        allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
        allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
        alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')
        allticlTracksterLinks = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTracksterLinks')

        simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
        tracksters = alltracksters.arrays(tsKeys)
        associations = allassociations.arrays(assKeys)
        tracks = alltracks.arrays(tracksKeys)
        tracksterLinks = allticlTracksterLinks.arrays(tsKeys + ['clue3DIndicesInTs'])

        # Extract track values (per event arrays)
        tracks_time = tracks.track_time_mtd
        tracks_timeErr = tracks.track_time_mtd_err
        tracks_MTDposX = tracks['track_pos_mtd/track_pos_mtd.theVector.theX']
        tracks_MTDposY = tracks['track_pos_mtd/track_pos_mtd.theVector.theY']
        tracks_MTDposZ = tracks['track_pos_mtd/track_pos_mtd.theVector.theZ']

        print(f"Processing {len(simtrackstersSC)} events...")
        for ev in tqdm(range(len(simtrackstersSC))):

            stsSCEv = simtrackstersSC[ev]
            tracksEv = tracks[ev]
            tsLinksEv = tracksterLinks[ev]
            assEv = associations[ev]

            tsEta = tsLinksEv.barycenter_eta
            tsPhi = tsLinksEv.barycenter_phi
            tsEnergy = tsLinksEv.raw_energy
            tsTime = tsLinksEv.time
            tsTimeError = tsLinksEv.timeError
            tsX = tsLinksEv.barycenter_x
            tsY = tsLinksEv.barycenter_y
            tsZ = tsLinksEv.barycenter_z

            dx = tsX[:, None] - tsX[None, :]
            dy = tsY[:, None] - tsY[None, :]
            dz = tsZ[:, None] - tsZ[None, :]

            distance_mat = np.sqrt(dx**2 + dy**2 + dz**2)
            tof_mat = distance_mat / C_CM_PER_NS

            valid = tsTimeError > 0
            valid_mat = valid[:, None] & valid[None, :]

            delta_t_mat = np.abs(tsTime[:, None] - tsTime[None, :]) - tof_mat
            delta_t_mat = np.where(valid_mat, delta_t_mat, 0.0)

            sigma_mat = np.sqrt(
                tsTimeError[:, None]**2 +
                tsTimeError[None, :]**2
            )

            pull_mat = delta_t_mat / sigma_mat

            mask_upper = np.triu(np.ones_like(pull_mat, dtype=bool), k=1)

            pull_pairs = pull_mat[mask_upper]
            delta_t_pairs = delta_t_mat[mask_upper]

            CLUE3Dprob = tracksters.id_probabilities[ev]
            tsLinksId = []
            for j, ids in enumerate(tsLinksEv.clue3DIndicesInTs):
                if len(ids) >1:
                    valid_ids = ids[ids<len(CLUE3Dprob)]
                    pids = np.argmax(CLUE3Dprob[valid_ids], axis=1)
                    tsLinksId.append(decide_pid(pids))
                else:
                    if ids[0]<len(CLUE3Dprob):
                        tsLinksId.append(np.argmax(CLUE3Dprob[ids[0]]))
                    else:
                        tsLinksId.append(0 if np.abs(tsZ[j])<368 else 4)
            tsLinksId = np.asarray(tsLinksId)

            for idx in range(len(stsSCEv.trackIdx)):

                # SELECT ASSOCIATED TRACKSTERS
                maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.99
                tsIdxAll = assEv.ticlTracksterLinks_simToReco_SC[idx]
                tsEnergyAssoc = tsLinksEv.raw_energy[tsIdxAll]
                sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
                maskEnergy = sharedEnergy / tsEnergyAssoc > 0.4

                mask = maskScore & maskEnergy
                assIndices = tsIdxAll[mask]

                if len(assIndices) == 0:
                    continue

                # OTHER (NEGATIVE) TRACKSTERS SAME SIDE
                n_ts = len(tsEta)
                assoc_mask = np.ones(n_ts, dtype=bool)
                assoc_mask[assIndices] = False

                sameSide = tsEta * stsSCEv.barycenter_eta[idx] > 0
                other_mask = assoc_mask & sameSide
                otherIndices = np.where(other_mask)[0]
                # ---------------------------------------------------------
                # TRACKSTER–TRACKSTER EDGES
                # ---------------------------------------------------------
                assIndices = np.asarray(assIndices)
                dT_ts_ass = []
                if len(assIndices) > 1:
                    dT_ts_ass = delta_t_mat[np.ix_(assIndices, assIndices)]

                otherIndices = np.asarray(otherIndices)
                dT_ts_other = []
                if len(assIndices) > 0 and len(otherIndices) > 0:
                    delta_t_mat_np = ak.to_numpy(delta_t_mat)
                    dT_ts_oth = delta_t_mat_np[np.ix_(assIndices, otherIndices)]

                pid_ass = []
                if len(assIndices) > 1:
                    ids_A = tsLinksId[assIndices]
                    pid_ass = ids_A[:, None] == ids_A[None, :]

                pid_other = []
                if len(assIndices) > 0 and len(otherIndices) > 0:
                    ids_A = tsLinksId[assIndices]
                    ids_B = tsLinksId[otherIndices]
                    pid_oth = (ids_A[:, None] == ids_B[None, :])

                # ----- associated pairs (positive)
                if len(assIndices) > 1:
                    for i in range(len(assIndices)):
                        for j in range(i+1, len(assIndices)):
                            idx1 = assIndices[i]
                            idx2 = assIndices[j]
                            feats = compute_ts_ts_edge_features(
                                tsEta[idx1], tsPhi[idx1], tsEnergy[idx1], tsZ[idx1], tsTime[idx1], tsTimeError[idx1],
                                tsEta[idx2], tsPhi[idx2], tsEnergy[idx2], tsZ[idx2], tsTime[idx2], tsTimeError[idx2],
                                dT_ts_ass[i][j], pid_ass[i][j]
                            )

                            all_ts_features.extend(feats)
                            all_ts_labels.extend(np.array([1], dtype=int))

                # ----- associated vs other (negative)
                if len(assIndices) > 0 and len(otherIndices) > 0:
                    for i in range(len(assIndices)):
                        for j in range(len(otherIndices)):
                            idx1 = assIndices[i]
                            idx2 = otherIndices[j]
                            feats = compute_ts_ts_edge_features(
                                tsEta[idx1], tsPhi[idx1], tsEnergy[idx1], tsZ[idx1], tsTime[idx1], tsTimeError[idx1],
                                tsEta[idx2], tsPhi[idx2], tsEnergy[idx2], tsZ[idx2], tsTime[idx2], tsTimeError[idx2],
                                dT_ts_oth[i][j], pid_oth[i][j]
                            )

                            all_ts_features.extend(feats)
                            all_ts_labels.extend(np.array([0], dtype=int))

                # GET TRACK INFO
                trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
                if trk_id == -1:
                    continue

                refPt = tracksEv.track_hgcal_pt[trk_id]
                refP = tracksEv.track_p[trk_id]
                refEta = tracksEv.track_hgcal_eta[trk_id]
                refPhi = tracksEv.track_hgcal_phi[trk_id]

                trk_time = tracks_time[ev][trk_id]
                trk_timeErr = tracks_timeErr[ev][trk_id]

                # ---------------------------------------------------------
                # TRACK–TRACKSTER EDGES
                # ---------------------------------------------------------
                trk_timeErr = tracks_timeErr[ev][trk_id]
                trk_time = tracks_time[ev][trk_id]
                trk_MTDposX = tracks_MTDposX[ev][trk_id]
                trk_MTDposY = tracks_MTDposY[ev][trk_id]
                trk_MTDposZ = tracks_MTDposZ[ev][trk_id]

                valid_time = (trk_timeErr > 0) & (tsTimeError > 0)

                tof = distance(trk_MTDposX, trk_MTDposY, trk_MTDposZ, tsX, tsY, tsZ) / C_CM_PER_NS
                delta_t = ak.where(valid_time, tsTime - trk_time - tof, 0.0)

                # ---- positive edges
                feats_pos = compute_track_ts_edge_features(
                    refPt, refP, refEta, refPhi,
                    trk_time, trk_timeErr,
                    tsEta[assIndices],
                    tsPhi[assIndices],
                    tsEnergy[assIndices],
                    tsTime[assIndices],
                    tsTimeError[assIndices],
                    delta_t[assIndices]
                )

                all_trk_features.extend(feats_pos)
                all_trk_labels.extend(np.ones(len(feats_pos), dtype=int))

                # ---- negative edges
                if len(otherIndices) > 0:

                    feats_neg = compute_track_ts_edge_features(
                        refPt, refP, refEta, refPhi,
                        trk_time, trk_timeErr,
                        tsEta[otherIndices],
                        tsPhi[otherIndices],
                        tsEnergy[otherIndices],
                        tsTime[otherIndices],
                        tsTimeError[otherIndices],
                        delta_t[other_mask]
                    )

                    all_trk_features.extend(feats_neg)
                    all_trk_labels.extend(np.zeros(len(feats_neg), dtype=int))

        file.close()

    # Convert to final arrays
    trk_features = np.asarray(all_trk_features, dtype=np.float32)
    trk_labels = np.asarray(all_trk_labels, dtype=np.int64)
    ts_features = np.asarray(all_ts_features, dtype=np.float32)
    ts_labels = np.asarray(all_ts_labels, dtype=np.int64)

    # Save compressed NPZ
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path,
                       trk_features=trk_features, trk_labels=trk_labels,
                       ts_features=ts_features, ts_labels=ts_labels)

    print(f"Saved shapes:")
    print(f"  Track-TS: features {trk_features.shape}, labels {trk_labels.shape}")
    print(f"  TS-TS:    features {ts_features.shape}, labels {ts_labels.shape}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare edge ML datasets from ROOT")
    parser.add_argument('--input', required=True, help="ROOT file(s)")
    parser.add_argument('--output', '-o', default='hgcal_linking_edges.npz', help="Output NPZ file")
    args = parser.parse_args()
    main(args)

