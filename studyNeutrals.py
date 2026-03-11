import sys
sys.path.append('..')

import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import mplhep as hep
import vector as vec
import matplotlib
from tqdm import tqdm
import math

from Timing.plotting import *

from numba import prange, njit
import awkward.numba

plt.style.use(hep.style.CMS)

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

print("START")

popt = np.array([ 1.49662496e+00,  4.04830636e+01, -1.19840947e+01, -4.04834456e+01,
  9.99984896e-01, -1.67329993e-03,  1.06828890e+01,  1.09862164e+00])

def normTracksters(x, y, a, b, c, d, e,f,  g, h):
    # x, y = pt ,eta
    y = np.abs(y)

    try:
        x = np.array(x)
        x = x.copy()
        x[x>200] = 200
    except:
        if x>200: x=200
    val = a + b*x + c*y + d*x**e + g * y**h  + f*x*y
    return val #np.clip(val, 0.008, 1)

all_neut_ass = []
all_neut_oth = []
all_neut_en_ass = []
all_neut_en_oth = []
all_neut_eta_ass = []
all_neut_eta_oth = []

plt.style.use(hep.style.CMS)

plt.figure(figsize=(20, 10))
plt.suptitle("Tracksters dR")
legend=True

def arr(x):
    return np.asarray(x, dtype=float)

for PT in [10, 50, 100, 200]:
    for ETA in [1.7, 2.2, 2.7]:
        label = "pt"+str(PT)+"_eta"+str(ETA).replace(".","p")
        file = uproot.open("/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo_"+label+"/histo_"+label+".root")

        print("opening file", label)

        #alltracksters = load_branch_with_highest_cycle(file,'ticlDumper/ticlTrackstersCLUE3DHigh')
        #allsimtrackstersCP = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
        allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
        allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
        #alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')
        allticlTracksterLinks = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTracksterLinks')

        #simtrackstersCP = allsimtrackstersCP.arrays(simTsKeys)
        simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
        # tracksters = alltracksters.arrays(tsKeys)
        associations = allassociations.arrays(assKeys)
        #tracks = alltracks.arrays(tracksKeys)
        tracksterLinks = allticlTracksterLinks.arrays(tsKeys)

        print("Compute residuals and pulls")

        allNeutral_dR_ass = []
        allNeutral_dR_other = []
        allNeutral_en_ass = []
        allNeutral_en_other = []
        allNeutral_eta_ass = []
        allNeutral_eta_other = []

        for ev in tqdm(prange(len(simtrackstersSC))):
            stsSCEv = simtrackstersSC[ev]
        #    tracksEv = tracks[ev]
            tsLinksEv = tracksterLinks[ev]
            assEv = associations[ev]
        #     print("--- ", ev, " ---")
            for idx in prange(len(stsSCEv.trackIdx)):
                ########## selecting good tracksters
                maskScore = assEv.ticlTracksterLinks_simToReco_SC_score[idx] < 0.9
                tsEnergy = tsLinksEv.raw_energy[assEv.ticlTracksterLinks_simToReco_SC[idx]]
                sharedEnergy = assEv.ticlTracksterLinks_simToReco_SC_sharedE[idx]
                maskEnergy = sharedEnergy / tsEnergy > 0.4
                maskIdx = maskScore & maskEnergy
                assIndices = assEv.ticlTracksterLinks_simToReco_SC[idx][maskIdx]
                sharedEnergy = sharedEnergy[maskScore & maskEnergy]
                ######### mask on the other tracksters
                n_ts = len(tsLinksEv.barycenter_eta)
                assoc_mask = np.ones(n_ts, dtype=bool)
                assoc_mask[assIndices] = False
                sameSide = tsLinksEv.barycenter_eta * stsSCEv.barycenter_eta[idx] > 0
                other_mask = sameSide & assoc_mask
                #########

                if not(len(assIndices)): continue # no associated tracksters

                allTsEta = tsLinksEv.barycenter_eta[other_mask]
                allTsPhi = tsLinksEv.barycenter_phi[other_mask]
                allTsZ = tsLinksEv.barycenter_z[other_mask]
                allTsEnergy = tsLinksEv.raw_energy[other_mask]

                tsEnergy = tsLinksEv.raw_energy[assIndices]
                ind = np.argsort(-tsEnergy)
                tsEnergy = tsEnergy[ind]
                shareTsEta = tsLinksEv.barycenter_eta[assIndices][ind]
                shareTsPhi = tsLinksEv.barycenter_phi[assIndices][ind]
                shareTsZ = tsLinksEv.barycenter_z[assIndices][ind]


                # find track position in the tracks array using track index
                dR_ass = []
                dR_other = []
                en_ass = []
                eta_ass = []
                en_oth = []
                eta_oth = []
                for i in range(len(shareTsEta)):
                    # compute distance from ts i to all ts j > i
                    dRs_a = distWrap_numba(
                        shareTsEta[i],
                        shareTsPhi[i],
                        shareTsEta[i+1:],
                        shareTsPhi[i+1:]
                    )
                    norm_a = normTracksters(np.ones(len(dRs_a))*tsEnergy[i], np.ones(len(dRs_a))*shareTsEta[i], *popt)
                    dR_ass.extend(dRs_a/norm_a)
                    en_ass.extend(np.ones(len(dRs_a))*tsEnergy[i])
                    eta_ass.extend(np.ones(len(dRs_a))*shareTsEta[i])
                    dRs = distWrap_numba(
                        shareTsEta[i],
                        shareTsPhi[i],
                        allTsEta,
                        allTsPhi
                    )
                    mask_oth_more_energetic = allTsEnergy > tsEnergy[i]
                    norm = normTracksters(np.where(mask_oth_more_energetic, allTsEnergy, tsEnergy[i]), np.where(mask_oth_more_energetic, allTsEta, shareTsEta[i]), *popt)
                    dR_other.extend(dRs /norm)
                    en_oth.extend(np.where(mask_oth_more_energetic, allTsEnergy, tsEnergy[i]))
                    eta_oth.extend(np.where(mask_oth_more_energetic, allTsEta, shareTsEta[i]))

                allNeutral_dR_ass.extend(dR_ass)
                allNeutral_dR_other.extend(dR_other)
                allNeutral_en_ass.extend(en_ass)
                allNeutral_en_other.extend(en_oth)
                allNeutral_eta_ass.extend(eta_ass)
                allNeutral_eta_other.extend(eta_oth)

        all_neut_ass.extend(allNeutral_dR_ass)
        all_neut_oth.extend(allNeutral_dR_other)
        all_neut_en_ass.extend(allNeutral_en_ass)
        all_neut_en_oth.extend(allNeutral_en_other)
        all_neut_eta_ass.extend(allNeutral_eta_ass)
        all_neut_eta_oth.extend(allNeutral_eta_other)

dR = arr(all_neut_ass)
eta = arr(all_neut_eta_ass)
E = arr(all_neut_en_ass)

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

plt.savefig("plots/neutrals/allTsdRnorm2_binned.png")

