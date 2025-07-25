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
# 'track_hgcal_x',
# 'track_hgcal_y',
# 'track_hgcal_z',
 'track_hgcal_eta',
 'track_hgcal_phi',
 'track_hgcal_pt',
 'track_pt',
 'track_hgcal_etaErr',
 'track_hgcal_phiErr',
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
# 'regressed_energy',
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
# 'ticlTracksterLinks_recoToSim_SC',
# 'ticlTracksterLinks_recoToSim_SC_score',
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
# 'time',
# 'timeError',
# 'regressed_energy',
 'raw_energy',
# 'raw_em_energy',
# 'raw_pt',
# 'raw_em_pt',
# 'barycenter_x',
# 'barycenter_y',
# 'barycenter_z',
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

path = '/eos/user/m/moanwar/TICLv5_samples/EnergyRegressionTICLv5PU_fromVertex/histo/SinglePions/'
files = os.listdir(path)

all_dataForNN = []

start = 0
for f in files:
    print(f"processing file {f}")
    filename = up.open(os.path.join(path,f))
    alltracksters = load_branch_with_highest_cycle(filename,'ticlDumper/ticlTrackstersCLUE3DHigh')
    allsimtrackstersSC = load_branch_with_highest_cycle(filename, 'ticlDumper/simtrackstersSC')
    allassociations = load_branch_with_highest_cycle(filename, 'ticlDumper/associations')
    alltracks = load_branch_with_highest_cycle(filename, 'ticlDumper/tracks')
    allticlTracksterLinks = load_branch_with_highest_cycle(filename, 'ticlDumper/ticlTracksterLinks')


    simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
    # tracksters = alltracksters.arrays(tsKeys)
    associations = allassociations.arrays(assKeys)
    tracks = alltracks.arrays(tracksKeys)
    tracksterLinks = allticlTracksterLinks.arrays(tsKeys)

    dataForNN = []

    for ev in tqdm(prange(len(simtrackstersSC))):
        stsSCEv = simtrackstersSC[ev]
        tracksEv = tracks[ev]
        tsLinksEv = tracksterLinks[ev]
        assEv = associations[ev]
    #     print(ev)
        for idx in prange(len(stsSCEv.trackIdx)):
            trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
            if trk_id == -1:
                continue

            refEta = tracksEv.track_hgcal_eta[trk_id]
            refPhi = tracksEv.track_hgcal_phi[trk_id]
            refPt = tracksEv.track_hgcal_pt[trk_id]
            refEtaErr = tracksEv.track_hgcal_etaErr[trk_id]
            refPhiErr = tracksEv.track_hgcal_phiErr[trk_id]
            refMissOut =tracksEv.track_missing_outer_hits[trk_id]
            #refPtErr = tracksEv.track_hgcal_ptErr[trk_id]

            ##########
            maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.8
            tsEnergy = tsLinksEv.raw_energy[assEv.ticlTracksterLinks_simToReco_SC[idx]]
            sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
            maskEnergy = sharedEnergy / tsEnergy > 0.4
            assIndices = assEv.ticlTracksterLinks_simToReco_SC[idx]
    #         print(sharedEnergy[:3], sharedEnergy[:3] / tsEnergy[:3])
            maskIdx = assIndices[maskScore & maskEnergy]
    #         good = []
    #         for recoIdx in maskIdx:
    #             idx_in_r2s = np.where(idx==assEv.ticlTracksterLinks_recoToSim_SC[recoIdx])[0][0]
    # #             print(np.where(idx==assEv.ticlTracksterLinks_recoToSim_SC[recoIdx])[0][0])
    #             if assEv.ticlTracksterLinks_recoToSim_SC_score[recoIdx][idx_in_r2s] < 0.6:
    #                 good.append(recoIdx)

    # #         print(f"{assIndices=}, {maskIdx=}, {good=}")
    #         #########

            assIndices = maskIdx

            allTsEta = tsLinksEv.barycenter_eta[tsLinksEv.barycenter_eta * refEta > 0]
            allTsPhi = tsLinksEv.barycenter_phi[tsLinksEv.barycenter_eta * refEta > 0]
            allTsEnergy = tsLinksEv.raw_energy[tsLinksEv.barycenter_eta * refEta > 0]
            shareTsEta = tsLinksEv.barycenter_eta[assIndices]
            shareTsPhi = tsLinksEv.barycenter_phi[assIndices]
            tsEnergy = tsLinksEv.raw_energy[assIndices]
            shareTsEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]

            if len(shareTsEta):
                d1 = distWrap_numba(refEta, refPhi, shareTsEta, shareTsPhi)
                idx_sort = np.array(d1).argsort()
                d1_sorted = d1[idx_sort]
                #distanceSharedList.append(d1_sorted)
                sharedEnergy_sorted = shareTsEnergy[idx_sort]
                fractionShared = np.cumsum(sharedEnergy_sorted) / np.sum(sharedEnergy_sorted)
                fractionPileup = np.cumsum(1. - (sharedEnergy_sorted / tsEnergy[idx_sort]))
                #distanceSharedList at which fractionSharedList == 1
                idx_R = np.where(fractionShared==1)[0][0] #CUT HERE! to be ==1 or <0.8
                dataForNN.append([refEta, refEtaErr, refPhi, refPhiErr, refPt, d1_sorted[idx_R], fractionPileup[idx_R]])
                #print(refEta, refPhi, refPt, d1_sorted[idx_R])
    all_dataForNN.extend(dataForNN)
    start+=1
    #if start == 10: break

#print(np.array(all_dataForNN).T)
all_dataForNN = np.array(all_dataForNN).T
np.savetxt("dataWithErr_cut1.txt", all_dataForNN)

# Extract the relevant columns
refEta = all_dataForNN[0]
refPhi = all_dataForNN[2]
refPt = all_dataForNN[4]
radius = all_dataForNN[5]
pileup = all_dataForNN[6]

# ----------------------------
# 1. Histograms: 2x2 subplot
# ----------------------------
#fig1, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
#plt.style.use(hep.style.CMS)
#axs = axs.flatten()
#
#axs[0].hist(refEta, bins=30, color='steelblue')
#axs[0].set_xlabel(r"$\eta$")
#axs[0].set_ylabel("counts")
#axs[0].set_title(r"$\eta$")
#
#axs[1].hist(refPt, bins=30, color='green')
#axs[0].set_xlabel(r"$p_T$")
#axs[0].set_ylabel("counts")
#axs[1].set_title(r"$p_T$")
#
#axs[2].hist(radius, bins=30, color='purple')
#axs[0].set_xlabel("radius")
#axs[0].set_ylabel("counts")
#axs[2].set_title(r"radius (in $\eta - \phi$)")
#
#axs[3].hist(pileup, bins=30, color='orange')
#axs[0].set_xlabel("pileup fraction")
#axs[0].set_ylabel("counts")
#axs[3].set_title("pileup contamination")
#
#for ax in axs:
#    ax.grid(True)
#
#plt.tight_layout()
#plt.savefig("histograms.png")
#
## --------------------------------------------
## 2. 2D Scatter Plots: 2x3 grid (6 relationships)
## --------------------------------------------
#fig2, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
#plt.style.use(hep.style.CMS)
#axs = axs.flatten()
#
#scatter_data = [
#    (refEta, radius, r"$\eta$", "radius", r"radius VS $\eta$"),
#    (refPt, radius, r"$p_T$", "radius", r"radius VS $p_T$"),
#    (refPhi, radius, r"$\phi$", "radius", r"radius VS $\phi$"),
#    (pileup, radius, "pileup", "radius", "radius VS pileup"),
#    (refEta, pileup, r"$\eta$", "pileup", r"pileup VS $\eta$"),
#    (refPt, pileup, r"$p_T$", "pileup", r"pileup VS $p_T$"),
#]
#
#for i, (x, y, xlabel, ylabel, title) in enumerate(scatter_data):
#    axs[i].scatter(x, y, alpha=0.7, s=10)
#    axs[i].set_xlabel(xlabel)
#    axs[i].set_ylabel(ylabel)
#    axs[i].set_title(title)
#    axs[i].grid(True)
#
#plt.tight_layout()
#plt.savefig("2Dplots.png")
