import os
os.environ['ABACUS'] = '/global/homes/b/boryanah/repos/abacus'
import sys
sys.path.append('/global/homes/b/boryanah/repos/abacus')

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import asdf

from Abacus import ReadAbacus
from abacusnbody.analysis import tsc
from abacusnbody.analysis import power_spectrum

import pynbody
import Pk_library as PKL

from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq

box = 2000.
Omega_M = 0.3175
InitialRedshift = 99.
growth = 1 / (1 + InitialRedshift)
H0 = 100.
c = 299792.458
f_NL = 100.
want_fixed_amp = False
amp_str = f"_notfixed" if not want_fixed_amp else ""
nmesh_ab = 256 #576
if nmesh_ab != 256:
    nmesh_str = f"_{nmesh_ab:d}"
else:
    nmesh_str = ""
    
read_abacus_kwargs = dict(
    pattern="ic_*",
    format='rvzel',
    add_grid=True,
    boxsize=box,
    dtype='f4',
)

ic_dir = "/global/cfs/cdirs/desicollab/users/lgarriso/zeldovich-fnl/"
if nmesh_ab == 576:
    ic_dir_ab = "/global/cfs/cdirs/desi/cosmosim/Abacus/ic/Abacus_pngbase_c302_ph000/"
    ic_dir_ab_fnl0 = "/global/cfs/cdirs/desi/cosmosim/Abacus/ic/Abacus_pngbase_c000_ph000/"
else:
    if want_fixed_amp:
        ic_dir_ab = "/global/cfs/cdirs/desicollab/users/lgarriso/zeldovich-fnl/quijote_png_LH_0_fixed/ic/"
        ic_dir_ab_fnl0 = "/global/cfs/cdirs/desicollab/users/lgarriso/zeldovich-fnl/quijote_png_LH_0_fixed/ic_fnl0/"
    else:
        ic_dir_ab = "/pscratch/sd/b/boryanah/PNG/ic/"
        ic_dir_ab_fnl0 = "/pscratch/sd/b/boryanah/PNG/ic_fnl0/"
        
# does it matter which transfer function we are using?
trans = np.loadtxt(ic_dir+'Tk_0.txt')
#trans = np.loadtxt("abacus_cosm000/CLASS_transfer")[:, :2]
trans[:, 1] /= trans[0, 1]

loginterp = CubicSpline(*np.log(trans).T)
trans_interp = lambda k: np.exp(loginterp(np.log(k)))

invM = lambda k: 3 * Omega_M * H0**2 / (2 * growth * c**2 * k**2 * trans_interp(k))

def get_pos(posdir, gadget=False):
    if gadget:
        pos = pynbody.load(posdir)
        pos['pos'] /= 1e3
    else:
        pos = ReadAbacus.from_dir(
            posdir,
            **read_abacus_kwargs,
            )
    return pos

def do_pk(posdir, gadget=False, nmesh=256, nthread=24):
    
    pos = get_pos(posdir, gadget=gadget)
    pk = power_spectrum.calc_power(
        pos['pos'],
        nmesh=nmesh,
        Lbox=box,
        nthread=nthread,
        compensated=True,
        interlaced=False,
        )
    return pk

def get_kgrid(N):
    return 2 * np.pi * np.stack(
        np.meshgrid(
            fftfreq(N, d=box/N), fftfreq(N, d=box/N), rfftfreq(N, d=box/N),
            indexing='ij',
            )
        )

def make_bardeen(delta):
    field_fft = rfftn(delta, overwrite_x=False, workers=24)
    # field_fft *= 1 / delta.size

    kgrid = get_kgrid(delta.shape[0])
    kmag = (kgrid**2).sum(axis=0)**0.5

    with np.errstate(divide='ignore', invalid='ignore'):
        field_fft *= invM(kmag)
    field_fft[0,0,0] = 0.

    delta = irfftn(field_fft, overwrite_x=True, workers=24)

    return delta


def do_bispectrum(posdir, gadget=False, nmesh=256, nthread=24, squeeze_kF=3, bardeen=True, fast=False):
    if nmesh != 576: #gadget: # B.H. cause we are not constructing delta
        pos = get_pos(posdir, gadget=gadget)
    
        delta = tsc.tsc_parallel(pos['pos'], nmesh, box, nthread=nthread)
        delta /= np.mean(delta, dtype=np.float64)
        delta -= 1.
    else:
        delta = asdf.open(posdir+f"ic_dens_N{nmesh}.asdf")['data']['density']
        

    if bardeen:
        delta = make_bardeen(delta)
    
    # pk = power_spectrum.calc_power(pos['pos'], **bk_kwargs)[1:]
    # theta = np.linspace(0, np.pi, 2)
    # theta = np.array([np.pi/3])
    kF = 2*np.pi/box
    ksamp = np.arange(squeeze_kF, 33 if fast else 65, 1)*kF
    # ksamp = [0.05, 0.1]
    res = []
    for k in ksamp:
        if fast:
            theta = np.array([2/3*np.pi])  # equilateral
        else:
            theta = np.array([2/3*np.pi, np.pi - 2*np.arcsin(squeeze_kF*kF/2/k)])  # equilateral, squeezed
        bk1 = PKL.Bk(delta, box, k, k, theta, 'TSC', threads=nthread)
        bk1 = vars(bk1)
        
        if not fast:
            theta = np.array([np.pi])  # folded
            bk2 = PKL.Bk(delta, box, k, k/2, theta, 'TSC', threads=nthread)
            bk2 = vars(bk2)

            # [k, k, k_eq, k_squeeze, k, k/2, k_folded]
            res += [{k: np.concatenate((bk1[k], bk2[k])) for k in bk1}]
        else:
            res += [bk1]
    res = {k: np.vstack([r[k] for r in res]).T for k in res[0].keys()}
    res['ksamp'] = ksamp
    res['squeeze_kF'] = squeeze_kF
    return res

fast = False

bk_abacus = do_bispectrum(ic_dir_ab, fast=fast, nmesh=nmesh_ab)
np.save(f"/pscratch/sd/b/boryanah/PNG/bk_abacus{nmesh_str}{amp_str}.npy", bk_abacus)

bk_abacus_fnl0 = do_bispectrum(ic_dir_ab_fnl0, fast=fast, nmesh=nmesh_ab)
np.save(f"/pscratch/sd/b/boryanah/PNG/bk_abacus_fnl0{nmesh_str}{amp_str}.npy", bk_abacus_fnl0)

if nmesh_ab != 576 and want_fixed_amp:
    bk_gadget = do_bispectrum(ic_dir+"quijote_png_LH_0_fixed/2LPTNGLC/ics", gadget=True, fast=fast)
    np.save("/pscratch/sd/b/boryanah/PNG/bk_gadget.npy", bk_gadget)

    bk_gadget_fnl0 = do_bispectrum(ic_dir+"quijote_png_LH_0_fixed/2LPTNGLC_fnl0/ics", gadget=True, fast=fast)
    np.save("/pscratch/sd/b/boryanah/PNG/bk_gadget_fnl0.npy", bk_gadget_fnl0)

    pk_abacus = do_pk(ic_dir_ab)
    np.save(f"/pscratch/sd/b/boryanah/PNG/pk_abacus.npy", pk_abacus)


def plot_bk_fast(bk_gadget, f_NL=-50., bardeen=True):
    fig, ax = plt.subplots(dpi=100)

    t = 0
    ax.plot(bk_gadget['ksamp'], bk_gadget['ksamp']**1 * (bk_gadget['B'][t] - bk_gadget_fnl0['B'][t])/f_NL, label='gadget')
    ax.plot(bk_gadget['ksamp'], bk_gadget['ksamp']**1 * primordial_deriv(bk_gadget, t=t, bardeen=bardeen), label='theory')
    # ax.plot(bk_gadget['ksamp'], (bk_gadget['B'][t] - bk_gadget_fnl0['B'][t])/f_NL/primordial_deriv(bk_gadget, t=t, bardeen=bardeen), label='gadget')
    ax.set_yscale('log')

    ax.legend()
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$k\frac{d}{df_{NL}} B(k,k,k)$')


def plot_bk(bk_gadget, bk_gadget_fnl0, bk_abacus, bk_abacus_fnl0, f_NL=-50., bardeen=True, fn=None):
    fig, axes = plt.subplots(3,1, sharex=True, gridspec_kw=dict(hspace=0), figsize=(6,8), dpi=100)

    tilt = 3
    for t,ax in enumerate(axes):
        ax.plot(bk_gadget['ksamp'], bk_gadget['ksamp']**tilt * (bk_gadget['B'][t] - bk_gadget_fnl0['B'][t])/f_NL, label='gadget')
        ax.plot(bk_abacus['ksamp'], bk_abacus['ksamp']**tilt * (bk_abacus['B'][t] - bk_abacus_fnl0['B'][t])/f_NL, label='abacus')
        if t == 2:
            ax.plot(bk_abacus['ksamp'], bk_abacus['ksamp']**tilt * primordial_deriv(bk_abacus, i=5, t=4, bardeen=bardeen), label='theory')
        else:
            ax.plot(bk_abacus['ksamp'], bk_abacus['ksamp']**tilt * primordial_deriv(bk_abacus, t=t, bardeen=bardeen), label='theory')

        ax.set_yscale('log')

    axes[0].legend()
    axes[-1].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[0].set_ylabel(r'$' + f'k^{tilt}' + r'\frac{d}{df_{NL}} B' + ('_\phi' if bardeen else '') + r'(k,k,k)$')
    axes[1].set_ylabel(r'$' + f'k^{tilt}' + r'\frac{d}{df_{NL}} B' + ('_\phi' if bardeen else '') + r'(k,k, )$') # +  f'{sqf if (sqf:=bk_abacus["squeeze_kF"]) > 1 else ""}' + r'k_F)$')
    axes[2].set_ylabel(r'$' + f'k^{tilt}' + r'\frac{d}{df_{NL}} B' + ('_\phi' if bardeen else '') + r'(k,k/2,k/2)$')

    if fn:
        fig.savefig(fn, bbox_inches='tight')


# t = 1 is squeezed; t = 4, i = 5, is k, k/2, k/2; t = 0, i = 1, is equi
def primordial_deriv_bardeen(bk, *, i=1, t=0):
    # for when the delta field is already the bardeen potential
    # bk['k']: [k, k, k_eq, k_squeeze, k, k/2, k_folded]
    # squeeze is k, k, 3kF so k k + k 3kF + k 3kF
    return 2 * (bk['Pk'][0] * bk['Pk'][i] + 
                bk['Pk'][0] * bk['Pk'][2+t] +
                bk['Pk'][i] * bk['Pk'][2+t])

def primordial_deriv(bk, *, i=1, t=0, bardeen=False):
    if bardeen:
        return primordial_deriv_bardeen(bk, i=i, t=t)
    # bk['k']: [k, k, k_eq, k_squeeze, k, k/2, k_folded]
    return 2 * (bk['Pk'][0] * bk['Pk'][i] * invM(bk['k'][0])**2 * invM(bk['k'][i])**2 + 
                bk['Pk'][0] * bk['Pk'][2+t] * invM(bk['k'][0])**2 * invM(bk['k'][2+t])**2 +
                bk['Pk'][i] * bk['Pk'][2+t] * invM(bk['k'][i])**2 * invM(bk['k'][2+t])**2 ) / \
                    (invM(bk['k'][0]) * invM(bk['k'][i]) * invM(bk['k'][2+t]))


plot_bk_fast(bk_gadget)

plot_bk(bk_gadget, bk_gadget_fnl0, bk_abacus, bk_abacus_fnl0, f_NL=f_NL, bardeen=True,
        # fn='bispectrum_validation.png',
        )
