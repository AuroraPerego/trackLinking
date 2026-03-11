import sys
sys.path.append('..')

if len(sys.argv) != 3:
    print("Usage: python3 studyEP.py PT ETA")
    sys.exit()

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

PT = int(sys.argv[1])
ETA = float(sys.argv[2])
label = "pt"+str(PT)+"_eta"+str(ETA).replace(".","p")
file = uproot.open("/eos/user/a/aperego/SampleProduction/TICLv5/ParticleGunPionPU/histo_"+label+"/histo_"+label+".root")

print("opening file", label)

alltracksters = load_branch_with_highest_cycle(file,'ticlDumper/ticlTrackstersCLUE3DHigh')
allsimtrackstersCP = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
allsimtrackstersSC = load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
allassociations = load_branch_with_highest_cycle(file, 'ticlDumper/associations')
alltracks = load_branch_with_highest_cycle(file, 'ticlDumper/tracks')
allticlTracksterLinks = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTracksterLinks')

simtrackstersCP = allsimtrackstersCP.arrays(simTsKeys)
simtrackstersSC = allsimtrackstersSC.arrays(simTsKeys)
# tracksters = alltracksters.arrays(tsKeys)
associations = allassociations.arrays(assKeys)
tracks = alltracks.arrays(tracksKeys)
tracksterLinks = allticlTracksterLinks.arrays(tsKeys)

print("Compute residuals and pulls")

#popt = np.array([2.29109869e-01, -1.95745285e+00,  4.97453548e+02,  1.95576520e+00,
#                -4.97619464e+02, -1.09398152e-04,  1.00016190e+00,  9.99776385e-01])
#
#def sigma_dR(xy, a, b, c, d, e, f, g, h):
#    x, y = xy  # pt, eta
#    y = np.abs(y)
#    val = a + b*x + c*y + d*x**g + e*y**h + f*x*y
#    #val = np.clip(val, 0.008, 0.1)
#    return val # np.nan_to_num(val, nan=0.05)

#popt = np.array([-2.85173951e-02, -4.96161458e+01, 4.18131615e-02,  4.96147581e+01,
#                  1.00000543e+00, -1.32656639e-04])
#
#def sigma_dR(xy, a, b, c, d, e, f):
#    x, y = xy # pt ,eta
#    y = np.abs(y)
#
#    try:
#        x = x.copy()
#        x[x>200] = 200
#    except:
#        if x>200: x=200
#    val = a + b*x + c*y + d*x**e + f*x*y #+ g * y**h
#    return np.clip(val, 0.008, 1)

popt = np.array([ 2.29516386e-01, -8.05371532e+02, -6.45573586e-01,  8.05370082e+02,
                  1.00000033e+00, -1.07458042e-04,  4.79631755e-01,  1.21330763e+00])

def sigma_dR(xy, a, b, c, d, e, f, g, h):
    x, y = xy # pt ,eta
    y = np.abs(y)

    try:
        x = x.copy()
        x[x>200] = 200
    except:
        if x>200: x=200
    try:
        y = y.copy()
        y[y<1.7] = 1.7
    except:
        if y<1.7 : y= 1.7
    val = a + b*x + c*y + d*x**e + f*x*y + g * y**h
    return np.clip(val, 0.008, 1)

plainEp_p_ass_all = []
plainEp_p_other_all = []

dR_ass_all = []
dR_other_all = []

pull_dir_ass_all = []
pull_dir_other_all = []

sigma_dR_all = []

for ev in tqdm(prange(len(simtrackstersSC))):
    stsSCEv = simtrackstersSC[ev]
    tracksEv = tracks[ev]
    tsLinksEv = tracksterLinks[ev]
    assEv = associations[ev]
#     print("--- ", ev, " ---")
    for idx in prange(len(stsSCEv.trackIdx)):
        # find track position in the tracks array using track index
        trk_id = find_track_id(tracksEv.track_id, stsSCEv.trackIdx[idx])
        if trk_id == -1:
            continue

        refEta = tracksEv.track_hgcal_eta[trk_id]
        refPhi = tracksEv.track_hgcal_phi[trk_id]
        refPt = tracksEv.track_hgcal_pt[trk_id]
        refP = tracksEv.track_p[trk_id]

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
        sameSide = tsLinksEv.barycenter_eta * refEta > 0
        other_mask = sameSide & assoc_mask
        #########

        if not(len(assIndices)): continue # no associated tracksters

        allTsEta = tsLinksEv.barycenter_eta[other_mask]
        allTsPhi = tsLinksEv.barycenter_phi[other_mask]
        allTsZ = tsLinksEv.barycenter_z[other_mask]
        allTsEnergy = tsLinksEv.raw_energy[other_mask]

        shareTsEta = tsLinksEv.barycenter_eta[assIndices]
        shareTsPhi = tsLinksEv.barycenter_phi[assIndices]
        shareTsZ = tsLinksEv.barycenter_z[assIndices]
        tsEnergy = tsLinksEv.raw_energy[assIndices]

        isEM_ass = np.abs(shareTsZ) < 368
        f_ass = np.where(isEM_ass, 1.0, 0.7)
        rEp_ass = (tsEnergy - f_ass * refP)
        plainEp_ass = (tsEnergy - refP)
        rEp_p_ass = (tsEnergy - f_ass * refP)/refP
        plainEp_p_ass = (tsEnergy - refP)/refP
        rEp_E_ass = (tsEnergy - f_ass * refP)/tsEnergy
        plainEp_E_ass = (tsEnergy - refP)/tsEnergy

        isEM_other = np.abs(allTsZ) < 368
        f_other = np.where(isEM_other, 1.0, 0.7)
        rEp_other = (allTsEnergy - f_other * refP)
        plainEp_other = (allTsEnergy - refP)
        rEp_p_other = (allTsEnergy - f_other * refP)/refP
        plainEp_p_other = (allTsEnergy - refP)/refP
        rEp_E_other = (allTsEnergy - f_other * refP)/allTsEnergy
        plainEp_E_other = (allTsEnergy - refP)/allTsEnergy

        dR_ass = distWrap_numba(refEta, refPhi, shareTsEta, shareTsPhi)
        dR_other = distWrap_numba(refEta, refPhi, allTsEta, allTsPhi)

        sigma_dir = sigma_dR((refPt, refEta), *popt)
        pull_dir_ass = dR_ass / sigma_dir
        pull_dir_other = dR_other / sigma_dir

        # E/p
        plainEp_p_ass_all.extend(ak.to_numpy(plainEp_p_ass))
        plainEp_p_other_all.extend(ak.to_numpy(plainEp_p_other))

        # Direction
        dR_ass_all.extend(ak.to_numpy(dR_ass))
        dR_other_all.extend(ak.to_numpy(dR_other))
        pull_dir_ass_all.extend(ak.to_numpy(pull_dir_ass))
        pull_dir_other_all.extend(ak.to_numpy(pull_dir_other))

        sigma_dR_all.append(sigma_dir)

def arr(x):
    return np.asarray(x, dtype=float)

plainEp_p_ass = arr(plainEp_p_ass_all)
plainEp_p_oth = arr(plainEp_p_other_all)

dR_ass = arr(dR_ass_all)
dR_other = arr(dR_other_all)
pull_dir_ass = arr(pull_dir_ass_all)
pull_dir_other = arr(pull_dir_other_all)
sigma = arr(sigma_dR_all)

plt.style.use(hep.style.CMS)
plt.figure(figsize=(30, 10))
plt.suptitle("dR")
plt.subplot(1, 3, 1)
plt.hist(dR_ass, bins=100, range=(0,0.2), density=True, label="associated")
plt.hist(dR_other, bins=100, range=(0,0.2), density=True, histtype="step", label="other")
plt.xlabel(r"$dR$")
plt.ylabel("Density")
plt.title("dR")
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
plt.hist(pull_dir_ass, bins=100, range=(0,4), density=True, label="associated")
plt.hist(pull_dir_other, bins=100, range=(0,4), density=True, histtype="step", label="other")
plt.xlabel(r"$dR/SCALE$")
plt.ylabel("Density")
plt.title("dR normalized")
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.hist(sigma, bins=100)
plt.xlabel(r"$\sigma_{dR}$")
plt.ylabel("Counts")
plt.title(f"Scale for normalization")
plt.grid()
plt.savefig("plots/dR/dR_"+label+"_new.png")

print("ROC and AUC")
from sklearn.metrics import roc_curve, auc

def build_score(wEp, rEp_ass_norm, rEp_oth_norm, pR_ass, pR_oth):
    wR = 1 - wEp
    score_ass = ( wEp * (rEp_ass_norm)**2 + wR * (pR_ass)**2 )
    score_oth = ( wEp * (rEp_oth_norm)**2 + wR * (pR_oth)**2 )

    y_true = np.concatenate([
        np.ones_like(score_ass),     # matched
        np.zeros_like(score_oth)     # unmatched
    ])

    y_score = np.concatenate([
        score_ass,
        score_oth
    ])

    fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def plot_ROC(fpr, tpr, roc_auc, wEp):
    plt.plot(fpr, tpr, label=f"AUC wEp="+ str(wEp) + f" = {roc_auc:.6f}")
    plt.plot([0,1], [0,1], "k--", lw=1)

wEp = 0.2
wR = 1 - wEp

plain_score_ass = ( wEp * (plainEp_p_ass)**2 + wR * (pull_dir_ass)**2 ) / 2
plain_score_oth = ( wEp * (plainEp_p_oth)**2 + wR * (pull_dir_other)**2 ) / 2

plt.style.use(hep.style.CMS)

plt.figure(figsize=(20, 10))
plt.suptitle("AUC and final cost")
plt.subplot(1, 2, 1)
for wEp in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    fpr, tpr, roc_auc = build_score(wEp, plainEp_p_ass, plainEp_p_oth, pull_dir_ass, pull_dir_other)
    plot_ROC(fpr, tpr, roc_auc, wEp)
plt.xlabel("False Positive Rate (unmatched)")
plt.ylabel("True Positive Rate (matched)")
plt.title("AUC score without f(p)")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
CUT = 5
n_ass = plt.hist(plain_score_ass, bins=100, range=(0,CUT), color = "dodgerblue", label=f"associated (over= {len(plain_score_ass[plain_score_ass>CUT])})")
n_oth = plt.hist(plain_score_oth, bins=100, range=(0,CUT), color = "red", histtype="step", label=f"other (over= {len(plain_score_oth[plain_score_oth>CUT])})")
THR=1
plt.text(CUT/4, max(max(n_ass[0]), max(n_oth[0]))*2/3, f"CUT at {THR}:\n{len(plain_score_ass[plain_score_ass<THR])/len(plain_score_ass)*100:.0f}% of ass ({len(plain_score_ass[plain_score_ass<THR])})\n"+
               f"{len(plain_score_oth[plain_score_oth<THR])/len(plain_score_oth)*100:.3f}% of others ({len(plain_score_oth[plain_score_oth<THR])})")
plt.xlabel("Score")
plt.ylabel("Counts")
plt.title("Cost wtihout f(p)")
plt.legend()
plt.grid()
plt.savefig("plots/dR/finalCost_"+label+"_new.png")

