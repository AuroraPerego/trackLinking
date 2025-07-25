import os
import awkward as ak
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
from tqdm import tqdm

from numba import prange, njit
import awkward.numba

plt.style.use(hep.style.CMS)

@njit
def flatten_numba(a):
    return [x[0] if len(x) else 0 for x in a]

def dist(refEta, refPhi, otherTsEta, otherTsPhi):
    return ((otherTsEta-refEta)**2 + (otherTsPhi-refPhi)**2)**0.5

@njit
def dist_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    out = []
    for i in range(len(otherTsEta)):
        distance = ((otherTsEta[i] - refEta) ** 2 + (otherTsPhi[i] - refPhi) ** 2) ** 0.5
        out.append(distance)
    return np.array(out)

@njit
def distWrap_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    out = []
    for i in range(len(otherTsEta)):
        deltaPhi = otherTsPhi[i] - refPhi
        deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
        distance = ((otherTsEta[i] - refEta) ** 2 + deltaPhi ** 2) ** 0.5
        out.append(distance)
    return np.array(out)

def find_track_id(array, number):
    try:
        return np.where(array == number)[0][0]
    except:
        return -1

def load_branch_with_highest_cycle(file, branch_name):
    # Get all keys in the file
    all_keys = file.keys()
    # Filter keys that match the specified branch name
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")
    # Find the key with the highest cycle
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
    # Load the branch with the highest cycle
    branch = file[highest_cycle_key]
    return branch

tracksKeys = [
 'track_id',
 'track_hgcal_x',
 'track_hgcal_y',
 'track_hgcal_z',
 'track_hgcal_eta',
 'track_hgcal_phi',
 'track_hgcal_pt',
 'track_pt',
 'track_missing_outer_hits',
#  'track_missing_inner_hits',
#  'track_quality',
#  'track_charge',
#  'track_time',
#  'track_time_quality',
#  'track_time_err',
#  'track_beta',
#  'track_time_mtd',
#  'track_time_mtd_err',
#  'track_pos_mtd',
#  'track_pos_mtd/track_pos_mtd.theVector.theX',
#  'track_pos_mtd/track_pos_mtd.theVector.theY',
#  'track_pos_mtd/track_pos_mtd.theVector.theZ',
#  'track_nhits',
#  'track_isMuon',
#  'track_isTrackerMuon'
]

simTsKeys = [
 'regressed_energy',
 'raw_energy',
 'trackIdx',
 # 'raw_em_energy',
 # 'raw_pt',
 # 'raw_em_pt',
 # 'barycenter_x',
 # 'barycenter_y',
 'barycenter_z',
 'barycenter_eta',
 'barycenter_phi',
 # 'EV1',
 # 'EV2',
 # 'EV3',
 # 'eVector0_x',
 # 'eVector0_y',
 # 'eVector0_z',
 # 'sigmaPCA1',
 # 'sigmaPCA2',
 # 'sigmaPCA3',
 # 'regressed_pt',
 # 'pdgID'
]

assKeys = [
 'ticlTracksterLinks_recoToSim_CP',
 'ticlTracksterLinks_recoToSim_CP_score',
#  'ticlTracksterLinks_recoToSim_CP_sharedE',
 'ticlTracksterLinks_simToReco_CP',
 'ticlTracksterLinks_simToReco_CP_score',
 'ticlTracksterLinks_simToReco_CP_sharedE',
#     'ticlCandidate_simToReco_CP_score',
#     'ticlCandidate_simToReco_CP_sharedE',
 'ticlTracksterLinks_recoToSim_SC',
 'ticlTracksterLinks_recoToSim_SC_score',
#  'ticlTracksterLinks_recoToSim_SC_sharedE',
 'ticlTracksterLinks_simToReco_SC',
 'ticlTracksterLinks_simToReco_SC_score',
 'ticlTracksterLinks_simToReco_SC_sharedE',
#     'ticlCandidate_simToReco_SC_score',
#     'ticlCandidate_simToReco_SC_sharedE'
          ]

tsKeys = [
#  'NTracksters',
#  'NClusters',
 'time',
 'timeError',
 'regressed_energy',
 'raw_energy',
 'raw_em_energy',
 'raw_pt',
 'raw_em_pt',
 'barycenter_x',
 'barycenter_y',
 'barycenter_z',
 'barycenter_eta',
 'barycenter_phi',
#  'EV1',
#  'EV2',
#  'EV3',
#  'eVector0_x',
#  'eVector0_y',
#  'eVector0_z',
#  'sigmaPCA1',
#  'sigmaPCA2',
#  'sigmaPCA3',
#  'id_probabilities',
#  'vertices_indexes',
#  'vertices_x',
#  'vertices_y',
#  'vertices_z',
#  'vertices_time',
#  'vertices_timeErr',
#  'vertices_energy',
#  'vertices_correctedEnergy',
#  'vertices_correctedEnergyUncertainty',
#  'vertices_multiplicity'
]


def process_file(f):
    print(f"processing file {f}")
    with up.open(os.path.join(path,f)) as filename:
        try:
#            allsimtrackstersCP = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
            allsimtrackstersSC = load_branch_with_highest_cycle(filename, 'ticlDumper/simtrackstersSC')
            allassociations = load_branch_with_highest_cycle(filename, 'ticlDumper/associations')
            alltracks = load_branch_with_highest_cycle(filename, 'ticlDumper/tracks')
            allticlTracksterLinks = load_branch_with_highest_cycle(filename, 'ticlDumper/ticlTracksterLinks')

#         simtrackstersCP = allsimtrackstersCP.arrays(simTsKeys)
            simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
            associations = allassociations.arrays(assKeys)
            tracks = alltracks.arrays(tracksKeys)
            tracksterLinks = allticlTracksterLinks.arrays(tsKeys)
        except:
            return []

    distanceSharedList = []
    distanceTotalList = []
    energySharedList = []
    energyTotalList = []
    fractionSharedList = []
    fractionTotalList = []
    fractionPileupList = []

    for ev in tqdm(prange(len(simtrackstersSC))):
        stsSCEv = simtrackstersSC[ev]
        tracksEv = tracks[ev]
        tsLinksEv = tracksterLinks[ev]
        assEv = associations[ev]
        for idx in prange(len(stsSCEv.trackIdx)):
            trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
            if trk_id == -1:
                continue

            refEta = tracksEv.track_hgcal_eta[trk_id]
            refPhi = tracksEv.track_hgcal_phi[trk_id]
            refPt = tracksEv.track_hgcal_pt[trk_id]

            ##########
            maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.9
            tsEnergy = tsLinksEv.raw_energy[assEv.ticlTracksterLinks_simToReco_SC[idx]]
            sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
            maskEnergy = sharedEnergy / tsEnergy > 0.4
            assIndices = assEv.ticlTracksterLinks_simToReco_SC[idx]
    #         print(sharedEnergy[:3], sharedEnergy[:3] / tsEnergy[:3])
            maskIdx = assIndices[maskScore & maskEnergy]
    #         good = []
    #         for recoIdx in maskIdx:
    #             idx_in_r2s = np.where(idx==assEv.ticlTracksterLinks_recoToSim_CP[recoIdx])[0][0]
    # #             print(np.where(idx==assEv.ticlTracksterLinks_recoToSim_CP[recoIdx])[0][0])
    #             if assEv.ticlTracksterLinks_recoToSim_CP_score[recoIdx][idx_in_r2s] < 0.6:
    #                 good.append(recoIdx)

    # #         print(f"{assIndices=}, {maskIdx=}, {good=}")
    #         #########

            assIndices = maskIdx
            sharedEnergy = sharedEnergy[maskScore & maskEnergy]

            allTsEta = tsLinksEv.barycenter_eta[tsLinksEv.barycenter_eta * refEta > 0]
            allTsPhi = tsLinksEv.barycenter_phi[tsLinksEv.barycenter_eta * refEta > 0]
            allTsEnergy = tsLinksEv.raw_energy[tsLinksEv.barycenter_eta * refEta > 0]
            shareTsEta = tsLinksEv.barycenter_eta[assIndices]
            shareTsPhi = tsLinksEv.barycenter_phi[assIndices]
            tsEnergy = tsLinksEv.raw_energy[assIndices]

            if len(shareTsEta):
                d1 = distWrap_numba(refEta, refPhi, shareTsEta, shareTsPhi)
                idx_sort = np.array(d1).argsort()
                d1_sorted = d1[idx_sort]
                distanceSharedList.append(d1_sorted)
                sharedEnergy_sorted = sharedEnergy[idx_sort]
                energySharedList.append(np.cumsum(sharedEnergy_sorted))
                fractionSharedList.append(np.cumsum(sharedEnergy_sorted) / np.sum(sharedEnergy_sorted))
                fractionPileupList.append(np.cumsum(1. - (sharedEnergy_sorted / tsEnergy[idx_sort])))
            else:
                continue

            d2 = distWrap_numba(refEta, refPhi, allTsEta, allTsPhi)
            idx2_sort = np.array(d2).argsort()
            d2_sorted = d2[idx2_sort]
            distanceTotalList.append(d2_sorted)
            allTsEnergy_sorted = allTsEnergy[idx2_sort]
            energyTotalList.append(np.cumsum(allTsEnergy_sorted))
            fractionTotalList.append(np.cumsum(allTsEnergy_sorted)/np.sum(sharedEnergy_sorted))

    return (distanceSharedList, distanceTotalList, energySharedList, energyTotalList, fractionSharedList, fractionTotalList, fractionPileupList)

if __name__ == "__main__":
    import multiprocessing as mp
    import sys

    if len(sys.argv)!=3:
        print("use this with two parameters: ./trackLinkingWindow.py pt eta")
        exit()

    pt = str(sys.argv[1]) #'10'
    eta = str(sys.argv[2]) #'1p7'
    path = f'/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo_pt{pt}_eta{eta}/'
    files = os.listdir(path) #[100:1000]
    files = [f for f in files if ".root" in f]

    all_distanceSharedList = []
    all_distanceTotalList = []
    all_energySharedList = []
    all_energyTotalList = []
    all_fractionSharedList = []
    all_fractionTotalList = []
    all_fractionPileupList = []
    batch_size = 10

    for i in range(0, len(files), batch_size):
        file_batch = files[i:i+batch_size]
        print(f"Processing files {i} to {i+len(file_batch)}...")

        with mp.Pool(processes=batch_size) as pool:
            res = list(tqdm(pool.imap(process_file, file_batch), total=len(file_batch)))

        for r1, r2, r3, r4, r5, r6, r7 in res:
            all_distanceSharedList.extend(r1)
            all_distanceTotalList.extend(r2)
            all_energySharedList.extend(r3)
            all_energyTotalList.extend(r4)
            all_fractionSharedList.extend(r5)
            all_fractionTotalList.extend(r6)
            all_fractionPileupList.extend(r7)

    #all_data= np.array(all_data).T
    print(f"Collected {len(all_distanceSharedList)} entries.")

    # plots
    MAX = 0.1
    NPOINTS = 100
    thresholds = np.linspace(0, MAX, NPOINTS)

    allEnergyShared = np.zeros(NPOINTS)
    allEnergyTotal = np.zeros(NPOINTS)
    allFractionShared = np.zeros(NPOINTS)
    allFractionTotal = np.zeros(NPOINTS)
    allFractionPileup = np.zeros(NPOINTS)

    def energy_at_thr(dist, energy, tr=np.linspace(0, MAX, NPOINTS)):
        idx = np.searchsorted(dist, tr, side="right") - 1
        arr = [energy[i] if i>=0 else 0 for i in idx]
        return np.array(arr)

    for distanceShare, energyShare, fractionShare, distanceAll, energyAll, fractionAll, pileup in zip(
        all_distanceSharedList, all_energySharedList, all_fractionSharedList, all_distanceTotalList, all_energyTotalList, all_fractionTotalList, all_fractionPileupList):

        # Aggregate totals
        allEnergyShared += energy_at_thr(distanceShare, energyShare)
        allEnergyTotal += energy_at_thr(distanceAll, energyAll)
        allFractionShared += energy_at_thr(distanceShare, fractionShare)
        allFractionTotal += energy_at_thr(distanceAll, fractionAll)
        allFractionPileup += energy_at_thr(distanceShare, pileup)

    allEnergyShared /= len(all_distanceSharedList)
    allEnergyTotal /= len(all_distanceSharedList)
    allFractionShared /= len(all_distanceSharedList)
    allFractionTotal /= len(all_distanceSharedList)
    allFractionPileup /= len(all_distanceSharedList)


    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(23, 8))
    fig.suptitle(f"Energy collected in the eta-phi window for pt {pt}, eta {eta}", y=0.93)
    plt.style.use(hep.style.CMS)

    ax1, ax2, ax3 = axs[0], axs[1], axs[2]

    ax1.plot(thresholds, allEnergyShared, marker='o', linestyle='-', label='shared')
    ax1.plot(thresholds, allEnergyTotal, marker='o', linestyle='-', label='total')
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Energy")
    ax1.set_title("Total and shared energy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(thresholds, allFractionShared, marker='o', linestyle='-', label='shared')
    ax2.plot(thresholds, allFractionTotal, marker='o', linestyle='-', label='total')
    ax2.axhline(0.8, color="red")
    ax2.axvline(thresholds[np.where(allFractionShared>0.8)[0][0]], color="red")
    ax2.text(thresholds[np.where(allFractionShared>0.8)[0][0]], 0.75, f" r = {thresholds[np.where(allFractionShared>0.8)[0][0]]:.3f}", ha='left', va='top')
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Fraction")
    ax2.set_title("Fractions of energy")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(thresholds, allFractionPileup, marker='o', linestyle='-', label='pileup')
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Fraction")
    ax3.set_title("Fraction of pileup")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"plotsWindow/cumulativeEnergyInWindow_200PU_pt{pt}_eta{eta}.png")
