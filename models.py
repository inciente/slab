import numpy as np; 
import xarray as xr; import pandas as pd; 
from datetime import datetime, timedelta

'''
Functions supporting slab models. 
Models are written as classes in the second half of this module.
'''

def refine_time( xr_obj, dt ):
    # Interpolate xr_obj onto a time vector
    # with predetermined dt (given in seconds)
    total_time = ( xr_obj['time'][-1] - xr_obj['time'][0] ).astype( \
                             'timedelta64[s]')/np.timedelta64(1,'s')
    nsteps = int( total_time / dt )
    new_time = [ pd.to_datetime( xr_obj['time'][0].values ) \
                  + timedelta( seconds = jj * dt ) for jj in range( nsteps ) ]
    return xr_obj.interp( time = new_time )

def drag_coeff( wspeed ):
    # Piece-wise to estimate drag coefficient based on wind speed
    Cd = 1.2e-3 * np.ones( wspeed.shape ); # ones
    faster = wspeed > 11; # meters per second
    Cd[faster] = ( 0.49 + 0.065 * wspeed.values[faster] ) * 1e-3
    fastest = wspeed > 25
    Cd[fastest] = 0.002;
    return Cd

def wstress( era_dat , surf = False):
    # Take in ERA5 data, including surface winds. 
    # Return wind stress vector (as complex number)    
    comp_vel = era_dat['u10'] + 1j*era_dat['v10']
    drag = drag_coeff( np.abs( comp_vel ) )
    stress = np.abs( comp_vel ) * drag * comp_vel * 1.22
    return stress

def vort_div( xr_obj ): 
    # compute vorticity and divergence from complex vector data
    divx = np.real( xr_obj ).differentiate( 'longitude' ) / ( 110e3 * np.cos( xr_obj['latitude'] / 180 * np.pi ) )
    divy = np.imag( xr_obj ).differentiate( 'latitude' ) / 110e3
    curl = np.imag( divx ) - np.real( divy )
    divv = np.real( divx ) + np.imag( divy )
    return curl, divv

def get_f( lat ):
    return 4 * np.pi * np.sin( lat / 180 * np.pi ) / 24 / 3600

# ---------------------------------
# ----- Time to write the models
# ---------------------------------


class double_decker:
    def __init__( self, ERA5 ):
        self.ERA5 = self.prepare_atm( ERA5 );

    def prepare_atm( self, era_dat ):
        era_dat['tau'] = wstress( era_dat )
        tau_curl, tau_div = vort_div( era_dat['tau'] )
        era_dat['G'] = tau_curl + 1j * tau_div
        return era_dat

    def empty( self ):
        vec = np.zeros( self.ERA5['time'].values.shape )
        vec = vec + 1j*vec.copy()
        return vec

    def initial_conditions( self, h0 ):
        variables = ['zeta', 'sigma', 'h', 'r' ]
        #data = xr.Dataset( )
        data = dict()
        for jj in variables:
            #data[ jj ] = xr.DataArray( data = self.empty(), 
            #                dims = ('time'),
            #                coords = { 'time': self.ERA5['time'].values } )
            data[jj] = self.empty()
        data['h'] = np.abs( data['h'] ); # delete imaginary
        data['h'][0] = h0
        return data


    def new_solution( self, x0, y0, h0 ):
        # Learn basic physical facts about the world
        f = get_f( y0 )
        location = { 'latitude':y0, 'longitude':x0 }
        forcing = self.ERA5.sel( location, method = 'nearest' )
        #time = self.ERA5['time'];
        #dt = pd.to_datetime( ( time[1] - time[0]).values , unit = 's' )

        # Solutions will start from zero.
        ml_sol = self.initial_conditions( h0 ) # dict for ML solution
        th_sol = self.initial_conditions( h0 ) # dict for thermocline
        
        # Generate a solver based on this information
        return decker_solver( ml_sol, th_sol, forcing, f )
        
class decker_solver:
    def __init__( self, ml_sol, th_sol, forcing, f ):
        self.ml_sol = ml_sol;
        self.th_sol = th_sol;
        self.forcing = forcing;
        self.f = f;
        self.U_storm = 8;
        self.time = forcing['time'].values
        self.dt = ( self.time[1] - self.time[0] ).astype( \
                             'timedelta64[s]')/np.timedelta64(1,'s')
        self.r = np.zeros( forcing['time'].shape ) + 0*1j

    def get_sol_index( self, sol_dict, tind ):
        # Extract values of solutions at given index or indices
        sol_at = dict()
        for key in sol_dict.keys():
            sol_at[ key ] = sol_dict[ key ][ tind ]
        return sol_at


    def apply_acceleration( self, to_obj, tind, accel_dict ):
        # Look through accel dictionary and apply
        # to determine solution of to_obj at time = tind + 1
        if to_obj == 'ML':
            sol_dict = self.ml_sol#self.get_sol_index( self.ml_sol, tind ) 
            #whole_sol = self.ml_sol.copy()
        else:
            sol_dict = self.th_sol; #now = self.get_sol_index( self.th_sol, tind )
            #whole_sol = self.th_sol.copy()

        sol_now = self.get_sol_index( sol_dict, tind )

        for var in accel_dict.keys():
            change = accel_dict[ var ] * self.dt 
            #print( change )
            #print( sol_now[var] + change )
            sol_dict[ var ][ tind + 1 ] = sol_now[ var ] + change
        
        #if to_obj == 'ML':
        #    self.ml_sol = whole_sol
        #else:
        #    self.th_sol = whole_sol

    def run_through( self ):
        for jj in range( len( self.time ) - 1 ):
            self.full_timestep( jj )

    def full_timestep( self, tind ):
        # First order slab model solution
        accels = {}
        accels['zeta'], accels['sigma'], accels['h'] \
                      = self.ML_first_order( tind )
        # Get second order correction
        accels = self.ML_second_order( accels, tind )
        self.apply_acceleration( 'ML' , tind, accels )
        # Thermocline solution
        accels = {}
        accels['zeta'], accels['sigma'], accels['h'] \
                      = self.TH_first_order( tind )
        self.apply_acceleration( 'TH' , tind, accels )
      
        # Second order correction to the mixed layer
        #accels = {} 
        #accels['zeta'], accels['sigma'] = self.ML_second_order( tind )
        #self.apply_acceleration( 'ML', tind, accels )


    def TH_energy( self, tind ):
        # Get current status of thermocline and mixed layer
        th_now = self.get_sol_index( self.th_sol, tind )
        ml_now = self.get_sol_index( self.ml_sol, tind )
        #th_now = self.th_sol.isel( time = tind )
        #ml_now = self.ml_sol.isel( time = tind )
        # Compute KE and PE of thermocline
        KE = 1024/2 * th_now['h'] * np.abs( th_now['zeta'] ) ** 2;
        PE = - 9.81/2 * th_now['h'] * ( th_now['h'] + 2 * ml_now['h'] )
        return KE

    def ML_first_order( self , tind ):
        # Apply first order balance to estimate acceleration 
        # of u, v, h, zeta, mu in the mixed layer, at time[tind]
        #ml_now = self.ml_sol.isel( time = tind )
        ml_now = self.get_sol_index( self.ml_sol, tind )
        forcing_now = self.forcing.isel( time = tind )
        
        dUdt = -1j * self.f * ml_now['zeta'] \
                  + forcing_now['tau'].values / 1024 / ml_now['h']
        dSdt = 1j * self.f * ml_now['sigma'] \
                  + forcing_now['G'].values / 1024 / ml_now['h']

        dhdt = - ml_now['h'] * np.imag( ml_now['sigma'] )

        return dUdt, dSdt, dhdt

    def ML_base_jerk( self, tind ):
        # Compute second time derivative of h_n
        if tind == 0:
            return 0
        else:
            #ml_use = self.ml_sol.isel( time = [tind-1,tind] )
            ml_use = self.get_sol_index( self.ml_sol, range( tind-1, tind+1 ) )
            dhdt = - ml_use['h'] * np.imag( ml_use['sigma'] )
            jerk = dhdt[1] - dhdt[0] 
            return jerk / self.dt


    def TH_first_order( self, tind ):
        # Apply first order solution for the thermocline 
        #ml_now = self.ml_sol.isel( time = tind )
        ml_now = self.get_sol_index( self.ml_sol, tind )
        #th_now = self.th_sol.isel( time = tind )
        th_now = self.get_sol_index( self.th_sol, tind )
        
        dUdt = self.U_storm * 1j * np.conjugate( th_now['sigma'] )
        dSidt = 1j * self.f * th_now['sigma'] 
        dhdt = - th_now['h'] * np.imag( th_now['sigma'] )
        
        if tind >= 1:
            # Calculate d2h/dt2
            buoyancy = 9.81 * ( 1023/1024 - 1 ); # reduced gravity
            add_accel = -1j * buoyancy / self.U_storm ** 2 
            dSidt += add_accel * self.ML_base_jerk( tind )
        
        return dUdt, dSidt, dhdt
    
    def wp_Eflux( self, tind ):
        # Compute pressure work (energy flux) across SML bottom
        ml_now = self.get_sol_index( self.ml_sol, tind )
        w = ml_now['h'] * np.imag( ml_now['sigma'] )
        # Compute pressure disturbance
        hprime = ml_now['h'] - self.ml_sol['h'][0] 
        Eflux = ( 9.81 * hprime * 1025 ) * ( 1024/1025 - 1 ) * w
        return Eflux

    def ML_second_order( self, accels, tind ):
        # Take in first order acceleartions and apply second order corrections
        ml_now = self.get_sol_index( self.ml_sol, tind )
        # Compute energy lost off bottom of SML via pressure work
        E_lost = self.wp_Eflux( tind ); 
        # COmpute total KE in mixed layer
        KE = ml_now['h'] * 1025 * np.abs( ml_now['zeta'] )**2 / 2 
        if KE == 0:
            r = 0 + 0*1j
        elif E_lost > 0:
            # energy transfer from ML to thermocline
            r = E_lost / KE;
        else:
            # energy transfer from thermocline to ML
            r = 0        

        self.r[tind] = r; 
        dUdt = - r * ml_now['zeta']; 
        dSdt = - r * ml_now['sigma']; 
        # Apply these corrections to acceleration dictionary
        accels['zeta'] += dUdt
        accels['sigma'] += dSdt
        return accels
