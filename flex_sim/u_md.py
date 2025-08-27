#from xtb.ase.calculator import XTB
from ase.optimize.lbfgs import LBFGS
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
import os
import time
import numpy as np
import pylab as pl
from ase.io import read, write
from ase.md import MDLogger
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen

def simpleMD(init_conf, temp, calc, fname, s, T):
    """
    Run a simple MD simulation with the given parameters.
    
    Parameters:
    init_conf: Initial configuration of the system.
    temp: Temperature in Kelvin.
    calc: Calculator to use for the simulation.
    fname: Filename for the trajectory.
    s: Interval for writing frames.
    T: Total time for the simulation.
    
    """
    
    init_conf.set_calculator(calc)

    #initialize the temperature

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp) #initialize temperature at 300
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=0.1) #drive system to desired temperature

    time_fs = []
    temperature = []
    energies = []

    #remove previously stored trajectory with the same name
    #os.system('rm -rfv '+fname)
    if os.path.exists(fname):
        os.remove(fname)
    print('removed previous trajectory file:', fname)

    fig, ax = pl.subplots(2, 1, figsize=(6,6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})

    def write_frame():
        dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
        dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
        dyn.atoms.write(fname, append=True)
        time_fs.append(dyn.get_time()/units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

        ax[0].plot(np.array(time_fs), np.array(energies), color="b")
        ax[0].set_ylabel('E (eV/atom)')

        # plot the temperature of the system as subplots
        ax[1].plot(np.array(time_fs), temperature, color="r")
        ax[1].set_ylabel('T (K)')
        ax[1].set_xlabel('Time (fs)')


    dyn.attach(write_frame, interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))
    return fig

def run_md_npt(
    init_conf,
    calculator,
    temperature,
    filename,
    nvt_time_fs,
    npt_time_fs,
    write_interval=10,
    pressure_bar=1.0
):  
    # 1. Setup 
    init_conf.set_calculator(calculator)

    # 2. Initialize velocities
    MaxwellBoltzmannDistribution(init_conf, temperature_K=temperature)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    # 3. === NVT equilibration ===
    timestep = 1.0 * units.fs
    output_prefix = filename.removeprefix('.xyz')

    dyn_nvt = Langevin(init_conf, timestep, temperature_K=temperature, friction=0.1)
    dyn_nvt.attach(lambda: init_conf.write(f"{output_prefix}.xyz", append=True), interval=write_interval)

    log_nvt = open(f"{output_prefix}_nvt.log", 'w')
    logger_nvt = MDLogger(dyn_nvt, init_conf, log_nvt, header=True, stress=False, peratom=False)
    dyn_nvt.attach(logger_nvt, interval=write_interval)

    print(f"Running NVT for {nvt_time_fs} steps...")
    t0 = time.time()
    dyn_nvt.run(nvt_time_fs)
    t1 = time.time()
    print(f"NVT finished in {(t1 - t0) / 60:.2f} minutes.")

    # 4. === NPT production ===


    dyn_npt = NPT(
        init_conf,
        timestep,
        temperature_K=temperature,
        externalstress=pressure_bar * units.bar,
        ttime=100.0 * units.fs,
        pfactor=3000.0 * units.fs**2 * units.eV
    )
    dyn_npt.attach(lambda: init_conf.write(f"{output_prefix}.xyz", append=True), interval=write_interval)

    log_npt = open(f"{output_prefix}_npt.log", 'w')
    logger_npt = MDLogger(dyn_npt, init_conf, log_npt, header=True, stress=True, peratom=False)
    dyn_npt.attach(logger_npt, interval=write_interval)

    print(f"Running NPT for {npt_time_fs} steps...")
    t0 = time.time()
    dyn_npt.run(npt_time_fs)
    t1 = time.time()
    print(f"NPT finished in {(t1 - t0) / 60:.2f} minutes.")


def MD(init_conf, temp, calc, fname, s, T):
    """
    Run a simple MD simulation with the given parameters.
    
    Parameters:
    init_conf: Initial configuration of the system.
    temp: Temperature in Kelvin.
    calc: Calculator to use for the simulation.
    fname: Filename for the trajectory.
    s: Interval for write the logfile, 20*s for writing frames.
    T: Total time for the simulation.
    
    """
    
    init_conf.set_calculator(calc)

    #initialize the temperature

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp) #initialize temperature at 300
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=1) #drive system to desired temperature

    dyn.attach(lambda: dyn.atoms.write(fname, append=True), interval=s*10)

    fname_log = fname.replace('.xyz', '.log')
    # 使用 ASE 提供的 logger 自动记录能量、温度、动能、总能等（每步写入）
    logfile = open(fname_log, 'w')
    logger = MDLogger(dyn, init_conf, logfile, header=True, stress=False, peratom=False)
    dyn.attach(logger, interval=s)  # 每步记录一次
    
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))

class XYZTrajectoryWriter:
    def __init__(self, atoms, filename="trajectory.xyz", interval=5):
        self.atoms = atoms
        self.filename = filename
        self.interval = interval
        self.counter = 0
        # Clear file at the start
        open(self.filename, 'w').close()

    def __call__(self):
        self.counter += 1
        if self.counter % self.interval == 0:
            write(self.filename, self.atoms, append=True)
            
def GeometryOptimization(atoms, outfile, logfile, interval=5, Fmax=0.01, stepMax=200, calc=None):
    """Run geometry optimization using XTB calculator.

    Args:
        atoms (ase.Atoms): Atoms object for optimization.
        outfile (str): Output filename for the trajectory.
        interval (int): Interval for writing frames.
        Fmax (float): Maximum force for convergence.
        stepMax (int): Maximum number of optimization steps.
    """
    if not calc:
        calc = XTB(method = 'GFN1-xTB')
    atoms.set_calculator(calc)
    opt = LBFGS(atoms, logfile=logfile)
    opt.attach(XYZTrajectoryWriter(atoms, filename=outfile, interval=interval))
    
    return opt.run(fmax=Fmax, steps=stepMax)


    