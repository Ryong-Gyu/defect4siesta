from NanoCore import *
import glob
import sys, os
import numpy as np
import copy
from mpmath import *
import matplotlib.pylab as plt
from scipy.optimize import fminbound, fsolve
from scipy.special import factorial



tol  = 1e-5


## Conversion parameters
bohr2ang   = 0.52918
hartree2eV = 27.2114
amu2emass  = 1822.89
kb         = 8.617 * 10**(-5)
smearing   = 0.050
e          = 2.71828
## Empirical parameters
gap = 1.516429
## Number of points
npt = 5



class luminescence(object):

    def __init__(self):

    # structure information
        ground   = sys.argv[1]
        excited  = sys.argv[2]
#        ground  = 'ground.fdf'
#        excited = 'excited.fdf'

        # read struct by NanoCore
        ground_struct   = s2.read_fdf(ground)
        excited_struct  = s2.read_fdf(excited)

        # ground state information
        ground_atoms     = ground_struct._atoms
        ground_cell      = ground_struct._cell
        ground_position  = [x._position for x in ground_atoms]
        ground_mass      = [m._mass for m in ground_atoms]
        # exicted state information
        excited_atoms    = excited_struct._atoms
        excited_cell     = excited_struct._cell
        excited_position = [x._position for x in excited_atoms]
        excited_mass     = [m._mass for m in excited_atoms]

        # store in namespaces
        self.ground_struct    = ground_struct
        self.ground_cell      = ground_cell
        self.ground_position  = np.array(ground_position,  dtype = float)
        self.ground_mass      = np.array(ground_mass,      dtype = float)

        self.excited_struct   = excited_struct
        self.excited_cell     = excited_cell
        self.excited_position = np.array(excited_position, dtype = float)
        self.excited_mass     = np.array(excited_mass,     dtype = float)
        self.init_info()

    def init_info(self):

        # Effective parameters
        mass        = self.ground_mass
        ground_pos  = self.ground_position
        excited_pos = self.excited_position

        dr, dr2, dR = self.deltaR(ground_pos,excited_pos)
        dQ = self.deltaQ(dr2, mass)  # total mass weighted distortions
        M  = self.modalM(dQ, dR)     # modal mass

        print('Calculation configuration coordinate diagram! \n')
        print('Total distortion:                %7.4f'%dR)
        print('Total mass weight distortion:    %7.4f'%dQ)


    def write_info(self):


        # get information from namespace
        dfc_g = self._dfc_g
        dfc_e = self._dfc_e

        dR = self._dR
        dQ = self._dQ
        M  = self._M

        hwg = self._hwg
        hwe = self._hwe
        wg  = self._wg
        we  = self._we

        Sg = self._Sg
        Se = self._Se

        Ezpl = self._ZPL
        Eact = self._Eact
        Eabs = self._Eabs
        Eems  = self._Eems

        dE = self._dE

        # export data for plot
        plotQ  = self._plotQ
        plotEg = self._plotEg
        plotEe = self._plotEe
        plotEc = self._plotEc
        plotQ1 = self._plotQ1
        plotQ2 = self._plotQ2
        plotQ3 = self._plotQ3
        plotE1 = self._plotE1
        plotE2 = self._plotE2
        plotE3 = self._plotE3

        print('Total distortion               dR:  %7.4f [ang]'%dR)
        print('Total mass weight distortion   dQ:  %7.4f [amu^1/2*ang]'%dQ)
        print('Modal mass                      M:  %7.4f [amu]'%M)
        print('Effective phonon energy (g)   hwg:  %7.4f [meV]'%hwg)       
        print('Effective phonon energy (e)   hwg:  %7.4f [meV]'%hwe)
        print('Zero phonon line             Ezpl:  %7.4f [eV]' %Ezpl)
        print('Activation energy            Eact:  %7.4f [eV]' %Eact)
        print('Activation energy (conduction)   :  %7.4f [eV]' %dE)
        print('Absortion energy             Eabs:  %7.4f [eV]' %Eabs)
        print('Emission energy             Eems:  %7.4f [eV]' %Eems)
        print('Huang-Rhys factor (g)           Sg:  %7.4f'%Sg)
        print('Huang-Rhys factor (e)           Se:  %7.4f'%Se)
        print('Franck-Cordon shift (g)           dfc_g:  %7.4f'%dfc_g)
        print('Franck-Cordon shift (e)           dfc_e:  %7.4f'%dfc_e)        


        # Effective parameters
        File = open('Effective_parameters.dat','w')
        File.write('Total distortion               dR:  %7.4f [ang]\n'%dR)
        File.write('Total mass weight distortion   dQ:  %7.4f [amu^1/2*ang]\n'%dQ)
        File.write('Modal mass                      M:  %7.4f [amu]\n'%M)
        File.write('Effective phonon energy (g)   hwg:  %7.4f [meV]\n'%hwg)
        File.write('Effective phonon energy (e)   hwg:  %7.4f [meV]\n'%hwe)
        File.write('Zero phonon line             Ezpl:  %7.4f [eV]\n' %Ezpl)
        File.write('Activation energy            Eact:  %7.4f [eV]\n' %Eact)
        File.write('Activation energy (conduction)   :  %7.4f [eV]\n' %dE)
        File.write('Absortion energy             Eabs:  %7.4f [eV]\n' %Eabs)
        File.write('Emission energy             Eems:  %7.4f [eV]\n' %Eems)
        File.write('Huang-Rhys factor (g)           Sg:  %7.4f\n'%Sg)
        File.write('Huang-Rhys factor (e)           Se:  %7.4f\n'%Se)
        File.write('Franck-Cordon shift (g)           dfc_g:  %7.4f\n'%dfc_g)
        File.write('Franck-Cordon shift (e)           dfc_e:  %7.4f\n'%dfc_e)
        File.close()


        # For plotting
        File2 = open('CCdiagram.dat','w')
        nQ = len(plotQ)
        for i in range(nQ):
            File2.write('%10.4f %10.4f %10.4f %10.4f\n'%(plotQ[i], plotEg[i], plotEe[i], plotEc[i]))
        File2.close()
        File3 = open('Ground.dat', 'w')
        nQ1 = len(plotQ1)
        for i in range(nQ1):
            File3.write('%10.4f %10.4f\n'%(plotQ1[i], plotE1[i]))
        File3.close()
        File4 = open('Excited.dat', 'w')
        nQ2 = len(plotQ2)
        for i in range(nQ2):
            File4.write('%10.4f %10.4f\n'%(plotQ2[i], plotE2[i]))
        File4.close()
        File5 = open('Conduction.dat', 'w')
        nQ3 = len(plotQ3)
        for i in range(nQ3):
            File5.write('%10.4f %10.4f\n'%(plotQ3[i], plotE3[i]))
        File5.close()



    def Polynomial2(self, parameters, x):

        A = parameters[0]
        B = parameters[1]
        C = parameters[2]
        Y = A*x**2 + B*x**1 + C
        return Y

    def Polynomial3(self, parameters, x):

        A = parameters[0]
        B = parameters[1]
        C = parameters[2]
        D = parameters[3]
        Y = A*x**3 + B*x**2 + C*x + D
        return Y


    def Polynomial4(self, parameters, x):

        A = parameters[0]
        B = parameters[1]
        C = parameters[2]
        D = parameters[3]
        E = parameters[4]
        Y = A*x**4 + B*x**3 + C*x**2 + D*x + E
        return Y




    def deltaR(self, pos1, pos2):

        dr  = pos2 - pos1
        dr2 = [sum(r**2) for r in dr]
        vector = sum(dr**2)
        print(vector)
        dr2 = np.array(dr2, dtype = float)
        dR  = np.sqrt(dr2.sum())

        for i in range(len(dr)):
            print((i+1))
            print(dr[i])

        return dr, dr2, dR

    def deltaCell(self, vector1, vector2):

        dCell = vector2 - vector1

        return dCell

    def deltaQ(self, dr2, mass):

        dq2  = [m*d for m, d in zip(dr2, mass)]
        dQ  = np.sqrt(sum(dq2))
        return dQ

    def modalM(self, dQ, dR):

        M   = (dQ/dR)**2
        return M

    def generate_struct(self):

        init_struct = self.ground_struct

        ground_position  = self.ground_position
        excited_position = self.excited_position

        dr, dr2, dR = self.deltaR(ground_position, excited_position)

        ddr   = dr/npt
        natom = len(ddr)

        for iddr in range(-2*npt,2*npt):
            struct = copy.copy(init_struct)
            for iatom in range(natom):
                pos = ground_position[iatom]
                pos2 = Vector(pos + iddr * ddr[iatom]) 
                struct._atoms[iatom].set_position(pos2)
            s2.Siesta(struct).write_struct()
            os.system('mv STRUCT.fdf linear%02d'%iddr)

    def generate_struct_neb(self):

        init_struct = self.ground_struct

        ground_position  = self.ground_position
        ground_cell      = self.ground_cell

        excited_position = self.excited_position
        excited_cell     = self.excited_cell

        dr, dr2, dR = self.deltaR(ground_position, excited_position)
        dCell       = self.deltaCell(ground_cell, excited_cell)

        ddr   = dr/npt
        ddCell = dCell/npt
        natom = len(ddr)

        for iddr in range(-2*npt,2*npt):
            struct = copy.copy(init_struct)
            for iatom in range(natom):
                pos = ground_position[iatom]
                pos2 = Vector(pos + iddr * ddr[iatom])
                struct._atoms[iatom].set_position(pos2)
            vector = ground_cell
            vector2 = vector + iddr * ddCell
            struct._cell = vector2
            s2.Siesta(struct).write_struct()
            os.system('mv STRUCT.fdf linear%02d'%iddr)

    def generate_system(self):

        os.system('mkdir ground_calc')        
        os.system('mkdir excited_calc')
        os.system('mkdir conduction_calc')

        struct_files = glob.glob('linear*')

#        for state in ['conduction']:
#        for state in ['excited']:
#        for state in ['ground']:
        for state in ['ground', 'excited','conduction']:
            os.chdir(state+'_calc')
            for istruct in struct_files:
                os.system('mkdir %s'%istruct)
                os.system('cp -r ../%s/input/ %s/.'%(state,istruct))
                os.system('cp ../%s %s/input/STRUCT.fdf'%(istruct,istruct))
                os.system('cp ../%s/slm* %s/.'%(state,istruct))
            os.chdir('..')

    def qsub_system(self):

        struct_files = glob.glob('linear*')

#        for state in ['conduction']:
#        for state in ['excited']:
#        for state in ['ground']:
        for state in ['ground', 'excited','conduction']:
            os.chdir(state+'_calc')
            for istruct in struct_files:
                os.chdir('%s'%istruct)
                if os.path.isdir('OUT'):
                    pass
                else:
                    os.system('sbatch slm*')
                os.chdir('..')
            os.chdir('..')

    def get_total_energy(self):

        struct_files = sorted(glob.glob('linear*'))

        ground_energy = []
        excited_energy = []
        conduction_energy = []

        q1 = []
        q2 = []
        q3 = []


        for state in ['ground', 'excited', 'conduction']:
            print(state)
            os.chdir(state+'_calc')
            for istruct in struct_files:
                print(istruct)

                if (os.path.isdir(istruct) != 1):
                    continue

                os.chdir('%s'%istruct)

                if os.path.isdir('OUT'):
                    os.chdir('OUT')


                    if state == 'ground':
                        try:
                            e = s2.get_total_energy()
                        except IndexError:
                            print('not converged')
                        else:

                            # collect the energy around the minimum
                            qtmp = float(istruct.split('linear')[-1])/npt
                            if (-1 - tol  <= qtmp and qtmp <= 1 + tol):
                                ground_energy.append(e)
                                q1.append(qtmp)


                    if state == 'conduction':
                        try:
                            e = s2.get_total_energy()
                        except IndexError:
                            print('not converged')
                        else:

                            # collect the energy around the minimum
                            qtmp = float(istruct.split('linear')[-1])/npt
                            if (-0.9 - tol  <= qtmp and qtmp <= 0 + tol):
                                conduction_energy.append(e)
                                q3.append(qtmp)


                    elif state == 'excited':
                        try:
                            e = s2.get_total_energy()
                        except IndexError:
                            print('not converged')
                        else:

                            # collect the energy around the minimum
                            qtmp = float(istruct.split('linear')[-1])/npt
                            if ( -0 - tol <= qtmp and qtmp <= 2 + tol):
                                excited_energy.append(e)
                                q2.append(qtmp)

                    os.chdir('..')
                os.chdir('..')
            os.chdir('..')        

        q1 = np.array(q1, dtype = float)
        q2 = np.array(q2, dtype = float)
        q3 = np.array(q3, dtype = float)


        print("Number of calcuated data (ground): %d"%len(q1))
        print("Number of calcuated data (exicid): %d"%len(q2))

        ground_energy = np.array(ground_energy, dtype = float)
        excited_energy = np.array(excited_energy, dtype = float) 
        conduction_energy = np.array(conduction_energy, dtype = float)


        return q1, q2, q3, ground_energy, excited_energy, conduction_energy


    def configurational_coordinate(self):

        q1, q2, q3, ground_energy, excited_energy, conduction_energy = self.get_total_energy()

        fit_q1 = q1
        fit_ground_energy = ground_energy
        fit_q2 = q2
        fit_excited_energy = excited_energy
        fit_q3 = q3
        fit_conduction_energy = conduction_energy

        parameter1 = plt.polyfit(fit_q1, fit_ground_energy, 2)
        parameter2 = plt.polyfit(fit_q2, fit_excited_energy, 2)
        parameter3 = plt.polyfit(fit_q3, fit_conduction_energy, 2)

        # function of polynomial
        ground_function  = lambda x : self.Polynomial2(parameter1,x) 
        excited_function = lambda x : self.Polynomial2(parameter2,x)
        conduction_function = lambda x : self.Polynomial2(parameter3,x)

        def func(x):
            return ground_function(x)-excited_function(x)
        # Conductation band to defect state
        def func2(x):
            return conduction_function(x) - excited_function(x)


        Qg = fminbound(ground_function,  min(q1), max(q1))
        Qe = fminbound(excited_function, min(q2), max(q2))
        Qc = fminbound(conduction_function, min(q3), max(q3))

        # Effective parameters
        mass        = self.ground_mass
        ground_pos  = self.ground_position
        excited_pos = self.excited_position

        dr, dr2, dR = self.deltaR(ground_pos,excited_pos)
        dQ = self.deltaQ(dr2, mass)  # total mass weighted distortions
        M  = self.modalM(dQ, dR)     # modal mass
        q  = fsolve(func, 1)

        q_conduction = fsolve(func2, 0)
        dE = excited_function(q_conduction) - conduction_function(Qc)

        dEg = ground_function(Qe) - ground_function(Qg)        
        dEe = excited_function(Qg) - excited_function(Qe)

        effective_phonon_energy_g = np.sqrt((2*dEg/hartree2eV)/(dQ**2/bohr2ang**2*amu2emass))*hartree2eV
        effective_phonon_energy_e = np.sqrt((2*dEe/hartree2eV)/(dQ**2/bohr2ang**2*amu2emass))*hartree2eV


        dhw = effective_phonon_energy_e - effective_phonon_energy_g
        activation_energy = excited_function(q) - excited_function(Qe)
        absortion_energy  = excited_function(Qg) - ground_function(Qg)
        emission_energy   = excited_function(Qe) - ground_function(Qe) 
        zero_phonon_line  = excited_function(Qe) - ground_function(Qg) + dhw
        binding_energy    = conduction_function(Qc) - excited_function(Qe)


        offset = dQ*(Qe-q)

        dfc_g = ground_function(Qe) - ground_function(Qg)
        dfc_e = excited_function(Qg) - excited_function(Qe)


        huang_rhys_factor_g  = dEg/effective_phonon_energy_g
        huang_rhys_factor_e  = dEe/effective_phonon_energy_e

        omega_ground  = np.sqrt(2*parameter1[0]/M)
        omega_excited = np.sqrt(2*parameter2[0]/M)


        # to evaluate tunneling effect  - 20.10.07 rong

        hw    = (effective_phonon_energy_g + effective_phonon_energy_e)/2
        theta = hw / (2 * 0.02568)
        S     = (huang_rhys_factor_e + huang_rhys_factor_g)/2
        p  = zero_phonon_line / hw
        if p >= S:
            x = S / p / np.sinh(theta)
        else:
            x = p / S / np.sinh(theta)
        y     = (1 + x**2) ** (0.5)

        tunneling_energy = zero_phonon_line / 2 * (
                           (y * np.cosh(theta) - x)/np.sinh(theta) -1)
        activation_energy2 = hw * (p - S)**2 / (4*S)


        ######visualization####

        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(111)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        #######################

        qfit = np.linspace(-20, 20, 300)
        ground_e  = self.Polynomial2(parameter1, qfit)
        excited_e = self.Polynomial2(parameter2, qfit)
        conduction_e = self.Polynomial2(parameter3, qfit)

        vbm = min(ground_energy)
        lv = min(excited_energy)
        print('potential barrier: %f \n'%(max(conduction_energy) - min(conduction_energy)))
        print('Binding energy: %f \n'%(binding_energy))

        size = 50
        plt.plot(qfit*dQ, ground_e -vbm,  color='k', linewidth=3.0, zorder=1)
        plt.plot(qfit*dQ, excited_e -vbm, color='b', linewidth=3.0, zorder=2)
#        plt.plot(qfit*dQ, conduction_e - vbm, color='r', linewidth=3.0, zorder=3)

        plt.scatter(q1*dQ, ground_energy -vbm, s = size, c = 'w', edgecolors='k', zorder=4, linewidths =2)
        plt.scatter(q2*dQ, excited_energy -vbm, s = size, c = 'w', edgecolors='b', zorder=5, linewidths =2)
        plt.scatter(q3*dQ, conduction_energy -vbm, s = size, c = 'w', edgecolors='r', zorder=6, linewidths =2)

#        plt.axhline(y=0, color='b', linestyle='-', linewidth=2.0)
#        plt.axhline(y=gap, color='r', linestyle='-', linewidth=2.0)

        plt.ylim(-1, 5)
        plt.xlim(-10,15)
        plt.xlabel(r'Q [amu$^{1/2} \AA$]', fontsize = 18)
        plt.ylabel('Total energy [eV]', fontsize =18)
        plt.tight_layout() 
        plt.savefig('ccdiagram.png')
#        plt.show()

        self._dR = dR
        self._dQ = dQ
        self._M  = M

        self._hwg = 1000*effective_phonon_energy_g
        self._hwe = 1000*effective_phonon_energy_e
        self._wg  = omega_ground
        self._we  = omega_excited

        self._Sg  = huang_rhys_factor_g
        self._Se  = huang_rhys_factor_e
        self._dEg  = dEg
        self._dEe  = dEe
        self._ZPL = zero_phonon_line
        self._Eabs = absortion_energy
        self._Eems = emission_energy
        self._Eact = activation_energy

        self._dfc_g =  dfc_g
        self._dfc_e =  dfc_e

        self._dE = dE

        # export plot
        self._plotQ = qfit*dQ 
        self._plotEg = ground_e - vbm
        self._plotEe = excited_e - vbm
        self._plotEc = conduction_e - vbm
        self._plotQ1 = q1*dQ
        self._plotQ2 = q2*dQ
        self._plotQ3 = q3*dQ
        self._plotE1 = ground_energy - vbm
        self._plotE2 = excited_energy - vbm
        self._plotE3 = conduction_energy - vbm



        self.write_info()


    def delta(self, x):

        result = np.exp(-(x/smearing)**2)/(smearing*np.sqrt(pi))
        return result

    def full_width_half_maxium(self, T, hwg, hwe, Sg, Se):

        hwg = 1/1000*self._hwg # eV
        hwe = 1/1000*self._hwe # eV
        Sg  = self._Sg
        Se  = self._Se

        if T == 0:
            WT = (np.sqrt(8*np.log(2))*Se*hwg/np.sqrt(Sg))
        elif T >0:
            WT = (np.sqrt(8*np.log(2))*Se*hwg/np.sqrt(Sg))*np.sqrt(coth(hwe/(2*kb*T)))

        return WT

    def dipole_transition(self, n, S):

        nfac = factorial(n, exact= False)

        return e**(-S)*S**(n)/nfac

    def L(self, hw, T, Ezpl, hwg, hwe, Sg, Se, n = 100):

        hwg = 1/1000*self._hwg # eV
        hwe = 1/1000*self._hwe # eV
        Sg  = self._Sg
        Se  = self._Se

        delta = np.vectorize(self.delta)

        l = 0
        wt = self.full_width_half_maxium(T, hwg, hwe, Sg, Se)
        for i in range(n):
            xe0xgn_2 = self.dipole_transition(i, Sg)
            d = delta(Ezpl-i*hwg-hw)
            l += wt * xe0xgn_2 * d
        return l
      
    def luminescence_lineshape(self):

        T   = 0
        Ezpl = self._ZPL
        hwg = 1/1000*self._hwg # eV
        hwe = 1/1000*self._hwe # eV
        Sg  = self._Sg
        Se  = self._Se

        hw = np.linspace(0, 3, 6000, dtype = float)
        l  = self.L(hw, T, Ezpl, hwg, hwe, Sg, Se)


        ######visualization####

        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        #######################

#        plt.axvline(x=Ezpl, color='b', linestyle='-', linewidth=2.0)
        plt.plot(hw, l, linewidth=2.0, color='b')
        plt.ylim(0,1.5)
        plt.xlim(0,2)
        plt.savefig('plintensity.png')
        plt.show()
        
    def absortion_profile(self):


        hwg = self._hwg/1000
        hwe = self._hwe/1000

        dfc_e = self._dfc_e

        self._dEg  = dEg
        self._dEe  = dEe
        self._ZPL = zero_phonon_line
        self._Eabs = absortion_energy
        self._Eems = emission_energy
        self._Eact = activation_energy

        self._dfc_g =  dfc_g
        self._dfc_e =  dfc_e



if __name__ == '__main__':

    test = luminescence()

    mode = input("(1)generate, qsub (2)generate, qsub(neb) (3)CC\n")
    mode = int(mode)

    if mode ==1:
        test.generate_struct()
        test.generate_system()
        test.qsub_system()

    elif mode ==2:
        test.generate_struct_neb()
        test.generate_system()
        test.qsub_system()

    elif mode ==3:
        test.configurational_coordinate()
#        test.luminescence_lineshape()
    else:
        print('wrong input yah~')
