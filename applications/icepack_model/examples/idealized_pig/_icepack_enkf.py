# ==============================================================================
# @des: This file contains run functions for icepack data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================

import numpy as np
import tqdm
import h5py

# --- import run_simulation function from the available examples ---
from ICESEE.applications.icepack_model.examples.idealized_pig._icepack_model import *
from ICESEE.config._utility_imports import icesee_get_index
from ICESEE.applications.icepack_model.icepack_utils._coordinates import (
    register_icepack_coordinate_provider,
)

register_icepack_coordinate_provider()


# --- Forecast step ---
def forecast_step_single(ensemble=None, **kwargs):
    """ensemble: packs the state variables:h,u,v of a single ensemble member
                 where h is thickness, u and v are the x and y components 
                 of the velocity field
    Returns: ensemble: updated ensemble member
    """
    #  call the run_model fun to push the state forward in time
    return run_model(ensemble, **kwargs)


# --- generate true state ---
def generate_true_state(**kwargs):
    """generate the true state of the model"""
    
    # # unpack the **kwargs
    smb  = kwargs.get('smb', None)
    basal_melt_field = kwargs.get('basal_melt_field', None)
    bed  = kwargs.get('bed', None)
    dt = kwargs.get('dt', None)
    nt = kwargs.get('nt', None)
    A0  = kwargs.get('A0', None)
    beta0  = kwargs.get('beta0', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u', None)
    s0 = kwargs.get('s0', None)
    floating = kwargs.get('floating', None)
    grounded = kwargs.get('grounded', None)
    solver = kwargs.get('solver', None)
    statevec_true = kwargs["statevec_true"]
    save_steps = kwargs.get('save_steps', None)

    # # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(**kwargs)

    
    # # --- fetch the state variables ---
    statevec_true[indx_map["h"],0] = h0.dat.data_ro
    statevec_true[indx_map["u"],0] = u0.dat.data_ro[:,0]
    statevec_true[indx_map["v"],0] = u0.dat.data_ro[:,1]
    statevec_true[indx_map["s"],0] = s0.dat.data_ro
    statevec_true[indx_map["basal_melt_field"],0] = basal_melt_field.dat.data_ro


    # # add BMR field to EnKF state vector if joint estimation is enabled 
    #if kwargs["joint_estimation"]:
        #statevec_true[indx_map["basal_melt_field"],0] = basal_melt_field.dat.data_ro

    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)
    basal_melt_field = basal_melt_field.copy(deepcopy=True)

    # --- extract a profile of the flowline at the initial state ---
    h_profiles, s_profiles, valid_points, distances, bed_values = initial_flowline_profile(kwargs)

    #### DEBUGGING
    print(f"\n INITIAL h_profiles_mean = {np.mean(h_profiles)}, s_profiles_mean = {np.mean(s_profiles)} \n")
    #############

    # --- step numbers at which to extract flowline profiles during the simulation -- 
    flowline_profile_steps = [t/dt for t in kwargs["save_steps"]]
    print(f"\n steps where profiles are sampled = {flowline_profile_steps} \n")

    hs_files = f"_modelrun_datasets/hs_profiles_true"
    
    with h5py.File(hs_files, "w") as F:
        dataset_h = F.create_dataset("h_profiles", (len(valid_points), len(save_steps) + 1), dtype = "f8")
        dataset_s = F.create_dataset("s_profiles", (len(valid_points), len(save_steps) + 1), dtype = "f8")
        dataset_s[:,0] = s_profiles # save the initial surface elevation profile
        dataset_h[:,0] = h_profiles # save the initial thickness profile
        dataset_bed = F.create_dataset("bed_values", data = bed_values)
        dataset_distances = F.create_dataset("distances", data = distances)

    
   
    kk = 0
    
    # loop through each step (k = 0 to k = 100)
    for k in range(nt):

        #step = k * dt
        step = k
        print(f"Time step = {step}")        

        ### Conditionals for depth-dependent basal melt rate function
        ### Select forcing scenario between 1935 - 2017 
        if step < (6/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 1935 - 1941
   
        elif (6 / dt) <= step < (15/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 1941 - 1950
    
        elif (15 / dt) <= step < (18/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 1950 - 1953
    
        elif (18/ dt) <= step < (20/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 1953 - 1955
        
        elif (20/ dt) <= step < (25/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 1955 - 1960
        
        elif (25/ dt) <= step < (27/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 1960 - 1962
        
        elif (27/ dt) <= step < (31/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 1962 - 1966
        
        elif (31/ dt) <= step < (40/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 1966 - 1975
        
        elif (40/ dt) <= step < (48/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 1975 - 1983
        
        elif (48/dt) <= step < (50/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 1983 - 1985
        
        elif (50/ dt) <= step < (59/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 1985 - 1994
        
        elif (59/ dt) <= step < (64/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 1994 - 2000
        
        elif (64/ dt) <= step < (69/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 2000 - 2005
        
        elif (69/ dt) <= step < (76/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "true") # 2005 - 2012
        
        elif (76/ dt) <= step:
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "true") # 2012 - 2017


        #### DEBUGGING 
        #print(f"\n Time step = {step}, h_mean = {np.mean(h.dat.data_ro)}, u_mean = {np.mean(u.dat.data_ro[:,0])}, v_mean = {np.mean(u.dat.data_ro[:,1])}, smb_mean = {np.mean(smb.dat.data_ro)}, bmr_mean = {np.mean(basal_melt_field.dat.data_ro)} \n")
        ###############
        


        # call the ice stream model to update the state variables
        h, u, s, floating, grounded = Icepack(solver, h, u, smb, basal_melt_field, bed, dt, h0, kwargs)

        statevec_true[indx_map["h"],k+1] = h.dat.data_ro
        statevec_true[indx_map["u"],k+1] = u.dat.data_ro[:,0]
        statevec_true[indx_map["v"],k+1] = u.dat.data_ro[:,1]
        statevec_true[indx_map["s"],k+1] = s.dat.data_ro
        statevec_true[indx_map["basal_melt_field"],k+1] = basal_melt_field.dat.data_ro


        # update the basal melt rate if joint estimation is enabled
        #if kwargs["joint_estimation"]:
            #statevec_true[indx_map["basal_melt_field"],k+1] = basal_melt_field.dat.data_ro
       
   
        if (kk <= len(flowline_profile_steps)-1):
            if (k == int(flowline_profile_steps[kk])):
                
                h_profiles, s_profiles = flowline_profile(h, s, valid_points)
                #### DEBUGGING
                print(f"\n Time step {k}: h_profiles_mean = {np.mean(h_profiles)}, s_profiles_mean = {np.mean(s_profiles)} \n")
                #############
                
                
                with h5py.File(hs_files, "a") as F:
                    F["h_profiles"][:,kk+1] = h_profiles
                    F["s_profiles"][:,kk+1] = s_profiles
                
                kk += 1

    updated_state = {}      
    for key in kwargs["vec_inputs"]:
        updated_state[key] = statevec_true[indx_map[key], :]

    return updated_state




# --- initialize the ensemble members ---
def initialize_ensemble(ens, **kwargs):
    
    """initialize the ensemble members"""

    ### unpack the **kwargs dictionary
    smb  = kwargs.get('smb', None)
    basal_melt_field = kwargs.get('basal_melt_field', None)
    bed  = kwargs.get('bed', None)
    dt = kwargs.get('dt', None)
    nt = kwargs.get('nt', None)
    A0  = kwargs.get('A0', None)
    beta0  = kwargs.get('beta0', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u', None)
    s0 = kwargs.get('s0', None)
    floating = kwargs.get('floating', None)
    grounded = kwargs.get('grounded', None)
    solver = kwargs.get('solver', None)
    save_steps = kwargs.get('save_steps', None)


    # # -- update parameter if joint estimation is enabled
    #if kwargs["joint_estimation"]:
        #basal_melt_field_nudged = basal_melt_field.dat.data_ro + kwargs["wrong_basal_melt_field"]
        #basal_melt_field_nudged = basal_melt_field.dat.data_ro
        #basal_melt_field = Function(Q)
        #basal_melt_field.dat.data[:] = basal_melt_field_nudged

    #h, u, s, floating, grounded = Icepack(solver, h, u0, smb, basal_melt_field, bed, dt, h0, kwargs)

    initialized_state = {'h': h0.dat.data_ro,
                         'u': u0.dat.data_ro[:,0], 
                         'v': u0.dat.data_ro[:,1],
                         's': s0.dat.data_ro,
                         'basal_melt_field': basal_melt_field.dat.data_ro}
    

    # --- create file to save ensemble flowline profiles ---
    h_profiles, s_profiles, valid_points, distances, bed_values = initial_flowline_profile(kwargs)
    hs_ensemble_files = f"_modelrun_datasets/hs_ensemble_profiles_ens{ens}_time0"


    with h5py.File(hs_ensemble_files, "w") as F:
    
        dataset_h_ensemble = F.create_dataset("h_profiles", len(valid_points), dtype = "f8")
        dataset_s_ensemble = F.create_dataset("s_profiles", len(valid_points), dtype = "f8")
        dataset_s_ensemble[:] = s_profiles 
        dataset_h_ensemble[:] = h_profiles 
        dataset_bed_ensemble = F.create_dataset("bed_values", data = bed_values)
        dataset_distances_ensemble = F.create_dataset("distances", data = distances)
        dataset_valid_points_ensemble = F.create_dataset("valid_points", data = valid_points)
    
    #print(f"\ndt = {dt}, h0_mean = {np.mean(h0.dat.data_ro)}, h_mean = {np.mean(initialized_state["h"])}, u_mean = {np.mean(u.dat.data_ro[:,0])}, v_mean = {np.mean(u.dat.data_ro[:,1])}\n")

    return initialized_state




# --- generate the nurged state ---
def generate_nurged_state(**kwargs):

    """generate the nudged state of the model"""
    
    # unpack the **kwargs
    smb  = kwargs.get('smb', None)
    basal_melt_field = kwargs.get('basal_melt_field', None)
    bed  = kwargs.get('bed', None)
    dt = kwargs.get('dt', None)
    nt = kwargs.get('nt', None)
    A0  = kwargs.get('A0', None)
    beta0  = kwargs.get('beta0', None)
    Q  = kwargs.get('Q', None)
    V  = kwargs.get('V', None)
    h0 = kwargs.get('h0', None)
    u0 = kwargs.get('u', None)
    s0 = kwargs.get('s0', None)
    floating = kwargs.get('floating', None)
    grounded = kwargs.get('grounded', None)
    solver = kwargs.get('solver', None)
    statevec_nurged = kwargs["statevec_nurged"]
    save_steps = kwargs.get('save_steps', None)
     

    # --- define the state variables list ---
    vec_inputs = kwargs["vec_inputs"]

    # call the icesee_get_index function to get the indices of the state variables
    vecs, indx_map, dim_per_proc = icesee_get_index(**kwargs)


    statevec_nurged[indx_map["h"],0]   = h0.dat.data_ro
    statevec_nurged[indx_map["u"],0]   = u0.dat.data_ro[:,0]
    statevec_nurged[indx_map["v"],0]   = u0.dat.data_ro[:,1]
    statevec_nurged[indx_map["s"],0]   = s0.dat.data_ro
    statevec_nurged[indx_map["basal_melt_field"],0] = basal_melt_field.dat.data_ro
    

    # # -- update parameter if joint estimation is enabled
    # if kwargs["joint_estimation"]:
    #     basal_melt_field_nudged = basal_melt_field.dat.data_ro * kwargs["wrong_basal_melt_field"]
    #     initial_perturbed_basal_melt_field = Function(Q)
    #     initial_perturbed_basal_melt_field.dat.data[:] = basal_melt_field_nudged
        
    #     statevec_nurged[indx_map["basal_melt_field"],0] = basal_melt_field_nudged

    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)
    basal_melt_field = basal_melt_field.copy(deepcopy=True)
         

    # --- extract a profile of the flowline at the initial state ---
    h_profiles, s_profiles, valid_points, distances, bed_values = initial_flowline_profile(kwargs)

    # --- step numbers at which to extract flowline profiles during the simulation -- 
    flowline_profile_steps = [t/dt for t in kwargs["save_steps"]]

    hs_nudged_files = f"_modelrun_datasets/hs_profiles_wrong"

    with h5py.File(hs_nudged_files, "w") as F:
        dataset_h = F.create_dataset("h_nudged_profiles", (len(valid_points), len(save_steps) + 1), dtype = "f8")
        dataset_s = F.create_dataset("s_nudged_profiles", (len(valid_points), len(save_steps) + 1), dtype = "f8")
        dataset_s[:,0] = s_profiles # save the initial surface elevation profile
        dataset_h[:,0] = h_profiles # save the initial thickness profile
        dataset_bed = F.create_dataset("bed_values", data = bed_values)
        dataset_distances = F.create_dataset("distances", data = distances)

    # # -- index for saving flowline profiles
    kk = 0
    

    # loop through each step (k = 0 to k = 100)
    for k in range(nt):

        step = k
        print(f"Time step = {step}") 

        ### Conditionals for depth-dependent basal melt rate function
        ### Select forcing scenario between 1935 - 2017 
        if step < (6/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 1935 - 1941
   
        elif (6 / dt) <= step < (15/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 1941 - 1950
    
        elif (15 / dt) <= step < (18/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 1950 - 1953
    
        elif (18/ dt) <= step < (20/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 1953 - 1955
        
        elif (20/ dt) <= step < (25/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 1955 - 1960
        
        elif (25/ dt) <= step < (27/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 1960 - 1962
        
        elif (27/ dt) <= step < (31/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 1962 - 1966
        
        elif (31/ dt) <= step < (40/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 1966 - 1975
        
        elif (40/ dt) <= step < (48/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 1975 - 1983
        
        elif (48/dt) <= step < (50/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 1983 - 1985
        
        elif (50/ dt) <= step < (59/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 1985 - 1994
        
        elif (59/ dt) <= step < (64/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 1994 - 2000
        
        elif (64/ dt) <= step < (69/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 2000 - 2005
        
        elif (69/ dt) <= step < (76/dt):
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='warm', experiment = "false") # 2005 - 2012
        
        elif (76/ dt) <= step:
            basal_melt_field, melt_max = BasalMeltRate(kwargs, step, floating, Q, s, h, scenario='control', experiment = "false") # 2012 - 2017

        
        # DEBUGGING
        print(f" \n at time step {step} maximum basal melt is {melt_max} \n ")
        
    
        
        # call the ice stream model to update the state variables
        h, u, s, floating, grounded = Icepack(solver, h, u, smb, basal_melt_field, bed, dt, h0, kwargs)

        statevec_nurged[indx_map["h"],k+1] = h.dat.data_ro
        statevec_nurged[indx_map["u"],k+1] = u.dat.data_ro[:,0]
        statevec_nurged[indx_map["v"],k+1] = u.dat.data_ro[:,1]
        statevec_nurged[indx_map["s"],k+1] = s.dat.data_ro
        statevec_nurged[indx_map["basal_melt_field"],k+1] = basal_melt_field.dat.data_ro


        # # -- update parameter if joint estimation is enabled
        #if kwargs["joint_estimation"]:
            #basal_melt_field_nudged = basal_melt_field.dat.data_ro * kwargs["wrong_basal_melt_field"]
            #basal_melt_field = Function(Q)
            #basal_melt_field.dat.data[:] = basal_melt_field_nudged

            #statevec_nurged[indx_map["basal_melt_field"],k+1] = basal_melt_field_nudged
        
        
       # # -- saving flowline profile at certain steps
        if (kk <= len(flowline_profile_steps)-1):
            if (k == int(flowline_profile_steps[kk])):
                
                h_profiles, s_profiles = flowline_profile(h, s, valid_points)
                print(s_profiles.shape,h_profiles.shape,"\n")
                #print(f"n\ average thickness = {np.mean(h.dat.data_ro)} and average BMR = {np.mean(basal_melt_field.dat.data_ro)} \n")
                
                
                with h5py.File(hs_nudged_files, "a") as F:
                    F["h_nudged_profiles"][:,kk+1] = h_profiles
                    F["s_nudged_profiles"][:,kk+1] = s_profiles
                
                kk += 1

    updated_state = {}      
    for key in kwargs["vec_inputs"]:
        updated_state[key] = statevec_nurged[indx_map[key], :]

    return updated_state
