import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import mplhep as hep
import vector as vec
import matplotlib
from tqdm import tqdm
import math

#from Timing.plotting import *

from numba import prange, njit
import awkward.numba

plt.style.use(hep.style.CMS)

@njit
def flatten_numba(a):
    return [x[0] if len(x) else 0 for x in a]

def dist(refEta, refPhi, otherTsEta, otherTsPhi):
    return ((otherTsEta-refEta)**2 + (otherTsPhi-refPhi)**2)**0.5

def distWrap2(refEta, refPhi, otherTsEta, otherTsPhi):
    deltaPhi = otherTsPhi - refPhi
    deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
    return ((otherTsEta - refEta) ** 2 + deltaPhi ** 2)

@njit
def distWrap2_numba(refEta, refPhi, otherTsEta, otherTsPhi):
    deltaPhi = otherTsPhi - refPhi
    deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi
    return ((otherTsEta - refEta) ** 2 + deltaPhi ** 2)

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

def distance(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5

C_CM_PER_NS = 29.9792458

##########
#  MAIN  #
##########

tracksKeys = [
 'track_id',
 'track_hgcal_x',
 'track_hgcal_y',
 'track_hgcal_z',
 'track_hgcal_eta',
 'track_hgcal_phi',
 'track_hgcal_pt',
 'track_pt',
 'track_p',
 'track_missing_outer_hits',
 'track_missing_inner_hits',
 'track_quality',
#  'track_charge',
#  'track_time',
#  'track_time_quality',
#  'track_time_err',
#  'track_beta',
 'track_time_mtd',
 'track_time_mtd_err',
#  'track_pos_mtd',
 'track_pos_mtd/track_pos_mtd.theVector.theX',
 'track_pos_mtd/track_pos_mtd.theVector.theY',
 'track_pos_mtd/track_pos_mtd.theVector.theZ',
 'track_nhits',
 'track_isMuon',
 'track_isTrackerMuon'
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
 #'trackTime',
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
 'pdgID'
]

assKeys = [
#  'ticlTracksterLinks_recoToSim_CP',
#  'ticlTracksterLinks_recoToSim_CP_score',
#  'ticlTracksterLinks_recoToSim_CP_sharedE',
 'ticlTracksterLinks_simToReco_SC',
 'ticlTracksterLinks_simToReco_SC_score',
 'ticlTracksterLinks_simToReco_SC_sharedE',
#     'ticlCandidate_simToReco_CP_score',
#     'ticlCandidate_simToReco_CP_sharedE'
#  'ticlCandidate_recoToSim_SC',
#  'ticlCandidate_recoToSim_SC_score',
#  'ticlCandidate_recoToSim_SC_sharedE',
#  'ticlCandidate_simToReco_SC',
#  'ticlCandidate_simToReco_SC_score',
#  'ticlCandidate_simToReco_SC_sharedE',
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

simcandkeys = [
 'simTICLCandidate_raw_energy',
 'simTICLCandidate_regressed_energy',
 # 'simTICLCandidate_simTracksterCPIndex',
 # 'simTICLCandidate_boundaryX',
 # 'simTICLCandidate_boundaryY',
 # 'simTICLCandidate_boundaryZ',
 # 'simTICLCandidate_boundaryPx',
 # 'simTICLCandidate_boundaryPy',
 # 'simTICLCandidate_boundaryPz',
 # 'simTICLCandidate_time',
 # 'simTICLCandidate_caloParticleMass',
 # 'simTICLCandidate_pdgId',
 # 'simTICLCandidate_charge',
 'simTICLCandidate_track_in_candidate']

candkeys = [
 # 'NCandidates',
 'candidate_charge',
 'candidate_pdgId',
 'candidate_id_probabilities',
 'candidate_time',
 'candidate_timeErr',
 'candidate_energy',
 'candidate_raw_energy',
 # 'candidate_px',
 # 'candidate_py',
 # 'candidate_pz',
 'track_in_candidate',
 'tracksters_in_candidate']

ALL_ass = []
ALL_oth = []
ALL_en_ass = []
ALL_en_oth = []
ALL_eta_ass = []
ALL_eta_oth = []

plt.style.use(hep.style.CMS)

plt.figure(figsize=(20, 10))
plt.suptitle("Tracksters dR")
legend=True

def resolution_model(X, A, B, C, D):
    E, eta = X
    val = (A + B*eta)/np.sqrt(E)+0.0001 + C + D*eta
    return np.clip(val, 0.05, 1)

#popt = np.array([ 0.10346782,  0.13112466, -0.0224533 ,  0.00476861])
popt = np.array([-0.34745398,  0.25698245,  0.00343885 ,-0.00204535])

def arr(x):
    return np.asarray(x, dtype=float)

for PT in [10, 50, 100, 200]:
    for ETA in [1.7, 2.2, 2.7]:
        label = "pt"+str(PT)+"_eta"+str(ETA).replace(".","p")
        file = uproot.open("/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo_"+label+"/histo_"+label+".root")

        print("opening file", label)

        allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
        alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')

        simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
        tracks = alltracks.arrays(tracksKeys)

        print("Compute residuals and pulls")

        all_dR_ass = []
        all_dR_other = []
        all_en_ass = []
        all_en_other = []
        all_eta_ass = []
        all_eta_other = []

        #pull_dir_ass_all = []
        #pull_dir_other_all = []

        #sigma_dR_all = []

        for ev in tqdm(prange(len(simtrackstersSC))):
            stsSCEv = simtrackstersSC[ev]
            tracksEv = tracks[ev]
            allTsEta = stsSCEv.barycenter_eta
            allTsPhi = stsSCEv.barycenter_phi

            for idx in prange(len(stsSCEv.trackIdx)):
                refTsEta = stsSCEv.barycenter_eta[idx]
                refTsPhi = stsSCEv.barycenter_phi[idx]
                refTsEnergy = stsSCEv.raw_energy[idx]

                sameSide = allTsEta * stsSCEv.barycenter_eta[idx] > 0
                closeMask = distWrap2(refTsEta, refTsPhi, allTsEta, allTsPhi) < 0.2**2
                other_mask = np.asarray(sameSide & closeMask)
                other_mask[idx] = 0

                if not(len(other_mask)): continue # nothing to do

                otherTsEta = allTsEta[other_mask]
                otherTsPhi = allTsPhi[other_mask]
                otherTsZ = stsSCEv.barycenter_z[other_mask]
                otherTsEnergy = stsSCEv.raw_energy[other_mask]

                # find track position in the tracks array using track index
                trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
                if trk_id == -1:
                    continue

                #charged tracksters
                refEta = tracksEv.track_hgcal_eta[trk_id]
                refPhi = tracksEv.track_hgcal_phi[trk_id]
                refPt = tracksEv.track_hgcal_pt[trk_id]
                refP = tracksEv.track_p[trk_id]

                dR_ass = distWrap2(refEta, refPhi, refTsEta, refTsPhi)**0.5
                #dR_other = distWrap_numba(refEta, refPhi, otherTsEta, otherTsPhi)
                norm = resolution_model((refP, refEta), *popt)

                all_dR_ass.append(dR_ass / norm)
                all_en_ass.append(refP)
                all_eta_ass.append(refEta)
                #try:
                #    all_dR_other.extend(dR_other / norm)
                #    all_en_other.extend(refP*len(dR_other))
                #    all_eta_other.extend(refEta*len(dR_other))
                #except:
                #    all_dR_other.append(dR_other / norm)
                #    all_en_other.append(refP*len(dR_other))
                #    all_eta_other.append(refEta*len(dR_other))

        ALL_ass.extend(all_dR_ass)
        #ALL_oth.extend(all_dR_other)
        ALL_en_ass.extend(all_en_ass)
        #ALL_en_oth.extend(all_en_other)
        ALL_eta_ass.extend(all_eta_ass)
        #ALL_eta_oth.extend(all_eta_other)

dR = arr(ALL_ass)
E = arr(ALL_en_ass)
eta = arr(ALL_eta_ass)

#import pandas as pd
#
#df = pd.DataFrame({
#        "dR": dR,
#        "E": E,
#        "eta": eta
#})
#
#df.to_csv("track_trackster_dR.csv", index=False)
#import sys
#sys.exit()

eta_bins = [(1.5, 1.95), (1.95, 2.45), (2.45, 3)]
eta_labels = [r"$\eta$ < 1.95", r"1.95 < $\eta$ < 2.45", r"$\eta$ > 2.45"]
energy_bins = [(0, 25), (25, 75), (75, 125), (125, np.inf)]
energy_labels = ["E < 25 GeV", "25 < E < 75 GeV", "75 < E < 125 GeV", "E > 125 GeV"]

fig, axes = plt.subplots(
    nrows=len(eta_bins),
    ncols=len(energy_bins),
    figsize=(10*len(energy_bins), 10*len(eta_bins)),
    sharex=True,
    sharey=True
)

for i, (eta_min, eta_max) in enumerate(eta_bins):
    eta_mask = (eta >= eta_min) & (eta < eta_max)

    for j, (E_min, E_max) in enumerate(energy_bins):
        ax = axes[i, j]

        energy_mask = (E >= E_min) & (E < E_max)
        mask = eta_mask & energy_mask

        dR_sel = dR[mask]

        if len(dR_sel) == 0:
            ax.text(0.5, 0.5, "No entries",
                    transform=ax.transAxes,
                    ha="center", va="center")
            continue

        # histogram
        ax.hist(
            dR_sel,
            bins=50,
            density=True,
            #histtype="stepfilled",
            color="dodgerblue",
            alpha=0.8
        )

        # quantiles
        q80 = np.quantile(dR_sel, 0.80)
        q90 = np.quantile(dR_sel, 0.90)

        ax.axvline(q80, color="red", linestyle="--", label=f"80%: {q80:.3f}")
        ax.axvline(q90, color="black", linestyle=":", label=f"90%: {q90:.3f}")

        ax.set_title(energy_labels[j] + " and " + eta_labels[i])
        if j == 0:
            ax.set_ylabel("Counts")
        if i == len(eta_bins) -1:
            ax.set_xlabel("dR norm")

        ax.set_xlim(-0.01, 1.01)
        ax.grid(True)
        ax.legend(loc="upper right")

plt.savefig("plots/dR/allSimTsdRnorm_binned2.png")
