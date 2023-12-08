use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::{epidemic::{Output, HealthStatus}, mobility::MobilityPars, utils::{retrieve_mobility, build_mobility_retriever_file_name, find_second_max_indices}, event::Event};


pub fn build_event_id_array(event_ensemble: &Vec<Event>, nlocs: usize, t_max: usize) -> Vec<Vec<Option<usize>>> {
    let mut event_id_lt = vec![vec![None; t_max]; nlocs];
    for (_event_idx, event) in event_ensemble.iter().enumerate() {
        let event_id = event.id as usize;
        let event_t = event.t as usize;
        let event_l = event.location as usize;
        event_id_lt[event_l][event_t] = Some(event_id);
    }
    event_id_lt
}

pub fn digest_raw_output(
    output_ensemble: &Vec<Output>, 
    a_vec: &Vec<f64>,
    mpars: &MobilityPars,  
    rho_hm: &HashMap<String, f64>,
    space_str: &String ) -> DigestedOutput {

    // Obtain effective number of locations
    let nlocs = a_vec.len();
    let a_cutoff = 0.000000001;
    let real_loc_indices: Vec<usize> = a_vec
    .iter()
    .enumerate()
    .filter(|(_, &x)| x > a_cutoff)
    .map(|(index, _)| index)
    .collect();
    let _nlocs_eff = real_loc_indices.len();

    let mob_str = build_mobility_retriever_file_name(&mpars, &rho_hm, &space_str);
    println!("{}", mob_str);
    let mobility_data = retrieve_mobility(&mob_str);
    let t_max = mobility_data[0].trajectory.len();

    // Binning setup
    let nbins = 31;
    let _rho_min = 0.0;
    let _rho_max = 1.0;

    let gamma = 0.21;

    let nsims = output_ensemble.len();

    // All output data structures needed
    let mut agents_per_rho = vec![vec![0; nbins]; nsims];
    let mut avg_foi_per_rho = vec![vec![0.0; nbins]; nsims];
    let mut avg_pc_foi_per_rho = vec![vec![0.0; nbins]; nsims];
    let mut avg_shared_per_rho = vec![vec![0.0; nbins]; nsims];
    let mut avg_size_per_rho = vec![vec![0.0; nbins]; nsims];
    let mut avg_t_pop_per_rho = vec![vec![0.0; nbins]; nsims];
    //let mut cum_exp_rate_per_rho = vec![vec![0.0; nbins]; nsims];
    let mut cum_i_pop_per_rho = vec![vec![0; nbins]; nsims];
    let mut cum_p_exp_per_rho = vec![vec![0.0; nbins]; nsims];
    let mut cum_shared_per_rho = vec![vec![0.0; nbins]; nsims]; 
    let mut cum_size_per_rho = vec![vec![0; nbins]; nsims];
    let mut cum_t_pop_per_rho = vec![vec![0; nbins]; nsims];
    let mut events_hh_per_rho = vec![vec![0; nbins]; nsims]; 
    let mut events_ho_per_rho = vec![vec![0; nbins]; nsims];
    let mut events_oh_per_rho = vec![vec![0; nbins]; nsims];
    let mut events_oo_per_rho = vec![vec![0; nbins]; nsims];
    let mut infected_per_rho = vec![vec![0; nbins]; nsims];
    let mut infected_h_per_rho = vec![vec![0; nbins]; nsims];
    let mut infected_o_per_rho = vec![vec![0; nbins]; nsims];
    let mut invaders_per_rho = vec![vec![0; nbins]; nsims];
    let mut nevents_eff_per_rho = vec![vec![0; nbins]; nsims];
    
    let mut nlocs_invaded = vec![0; nsims];

    let mut t_inv_dist_per_loc: Vec<Vec<u32>> = vec![vec![0; nlocs]; nsims];
    let mut t_peak_dist_per_loc: Vec<Vec<u32>> = vec![vec![0; nlocs]; nsims];
    let mut total_cases_per_loc = vec![vec![0; nlocs]; nsims];

    let mut a_exp_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 2]; nsims];
    let mut a_inf_h_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut a_inf_o_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut avg_a_h_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut avg_a_o_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut da_trip_hh_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut da_trip_ho_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut da_trip_oh_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut da_trip_oo_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_inf_h_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_inf_o_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_inf_tr_h_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_inf_tr_o_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_trip_hh_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_trip_ho_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_trip_oh_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut f_trip_oo_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut t_inf_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];
    let mut t_inv_dist_per_rho: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nbins]; nsims];

    let mut infected_rho_per_loc: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nlocs]; nsims];
    let mut infected_rho_h_per_loc: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nlocs]; nsims];
    let mut infected_rho_o_per_loc: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nlocs]; nsims];
    let mut invader_rho_per_loc: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nlocs]; nsims];

    let mut binned_stats_per_sim: Vec<BinnedStats> = vec![BinnedStats::new(nbins, nlocs); nsims];
    let mut event_data_per_sim: Vec<EventOutput> = vec![EventOutput::new(); nsims];

    let mut returner_netmob_per_sim: Vec<HashMap<(usize, usize), u32>> = vec![HashMap::new(); nsims];
    let mut commuter_netmob_per_sim: Vec<HashMap<(usize, usize), u32>> = vec![HashMap::new(); nsims];
    let mut explorer_netmob_per_sim: Vec<HashMap<(usize, usize), u32>> = vec![HashMap::new(); nsims]; 

    let infectious_period_cutoff = 0;

    // Loop over ensemble of simulations to compute & store data
    for (sim_idx, output) in output_ensemble.iter().enumerate() {
        println!("sim={sim_idx}");
        let agent_ensemble = output.agent.inner();
        let event_ensemble = output.event.inner(); 

        let event_id_matrix = build_event_id_array(&event_ensemble, nlocs, t_max+1);

        let mut infected_tseries = vec![vec![0; t_max+1]; nlocs];

        // Loop over locations to get info about the invasion process
        for loc in real_loc_indices.iter() {
            let event_id_t = &event_id_matrix[*loc];
            let nonnone_event_id_t: Vec<usize> = event_id_t.iter()
                .enumerate()
                .filter(|(_, &id)| id.is_some())
                .map(|(ind, _)| ind)
                .collect();

            // Was there an invasion at all?
            if nonnone_event_id_t.is_empty() == false {
                // Location was invaded: Obtain event index & rest of info
                let index = nonnone_event_id_t[0];
                let event_id = event_id_matrix[*loc][index].unwrap() as usize;    
                if let Some(event) = event_ensemble.iter().find(|&e| e.id == event_id as u32) {
                    let invader_rho = event.infector_rho; 

                    if invader_rho <= 1.0 {
                        let rho_idx = (invader_rho * (nbins as f64)) as usize;
                        let invasion_time = event.t as usize;
 
                        invaders_per_rho[sim_idx][rho_idx] += 1;
                        nlocs_invaded[sim_idx] += 1;
                        t_inv_dist_per_rho[sim_idx][rho_idx].push(invasion_time as f64);
                        invader_rho_per_loc[sim_idx][*loc].push(invader_rho);
                        t_inv_dist_per_loc[sim_idx][*loc] = invasion_time as u32;
                    }
                }
            }
        }

        // Loop over events
        for (_event_idx, event) in event_ensemble.iter().enumerate() {
            let loc = event.location;
            let inf_pop_avg_rho = event.inf_pop_avg_rho;
            let infector_rho = event.infector_rho;
            let size = event.size;
            let time = event.t;
            let attractiveness = a_vec[loc as usize];
            let tot_pop = event.total_population;

            event_data_per_sim[sim_idx].location.push(loc);
            event_data_per_sim[sim_idx].attractiveness.push(attractiveness);
            event_data_per_sim[sim_idx].inf_pop_avg_rho.push(inf_pop_avg_rho);
            event_data_per_sim[sim_idx].infector_rho.push(infector_rho);
            event_data_per_sim[sim_idx].size.push(size);
            event_data_per_sim[sim_idx].time.push(time);
            event_data_per_sim[sim_idx].tot_pop.push(tot_pop);
        }

        // Loop over agents to get general info about the infection & mobility process
        for (_agent_idx, agent) in agent_ensemble.iter().enumerate() {
            let mob_id = agent.mob_id as usize;
            let health = agent.status;
    
            let rho = agent.mobility;
            let rho_idx = (rho * (nbins as f64)) as usize;
            agents_per_rho[sim_idx][rho_idx] += 1;

            // Load trajectory & obtain home
            let mut trajectory = mobility_data[mob_id].trajectory.clone();
            let home = trajectory[0] as usize;

            if health == HealthStatus::Removed {

                let l_inf = agent.infected_where.unwrap() as usize;

                infected_rho_per_loc[sim_idx][l_inf].push(rho);
                total_cases_per_loc[sim_idx][l_inf] += 1;

                let visits = trajectory.iter().filter(|&&x| x == l_inf as u32).count();
                let inf_freq = visits as f64 / trajectory.len() as f64;

                let a_inf = a_vec[l_inf];

                let out_infection;
                if l_inf == home {
                    infected_rho_h_per_loc[sim_idx][l_inf].push(rho);
                    infected_h_per_rho[sim_idx][rho_idx] += 1;
                    a_inf_h_dist_per_rho[sim_idx][rho_idx].push(a_inf);
                    f_inf_h_dist_per_rho[sim_idx][rho_idx].push(inf_freq);
                    out_infection = false;
                } else {
                    infected_rho_o_per_loc[sim_idx][l_inf].push(rho);
                    a_inf_o_dist_per_rho[sim_idx][rho_idx].push(a_inf);
                    f_inf_o_dist_per_rho[sim_idx][rho_idx].push(inf_freq);
                    infected_o_per_rho[sim_idx][rho_idx] += 1;
                    out_infection = true;
                }

                let t_inf = agent.infected_when.unwrap() as usize;
                t_inf_dist_per_rho[sim_idx][rho_idx].push(t_inf as f64);
                let t_rem = agent.removed_when.unwrap() as usize;

                if t_rem - t_inf > infectious_period_cutoff {
                    infected_per_rho[sim_idx][rho_idx] += 1;
                }

                for i in (t_inf + 1)..(t_rem + 1) {
                    infected_tseries[l_inf][i] += 1;
                }

                let mut unique_values: HashSet<u32> = HashSet::new();
                let traj_before_inf = &trajectory[0..t_inf as usize];
                for &value in traj_before_inf {
                    unique_values.insert(value);
                }
                let unique_visits = unique_values.len();
                let p_exp = agent.mobility * f64::powf(unique_visits as f64, -gamma);
                cum_p_exp_per_rho[sim_idx][rho_idx] += p_exp;

                let mut nevents_eff = 0;
                let mut cum_a_h = 0.0;
                let mut cum_a_o = 0.0;
                let mut cum_shared = 0.0;
                let mut cum_size = 0;
                let mut cum_i_pop = 0;
                let mut cum_t_pop = 0;
                let mut cum_pc_foi = 0.0;
                let mut cum_foi = 0.0;

                let mut prev_loc;
                if t_inf == 0 {
                    prev_loc = home;
                } else if t_inf >= t_max {
                    prev_loc = trajectory[t_inf-1] as usize;
                    println!("Objection!");
                } else {
                    prev_loc = trajectory[t_inf-1] as usize;
                }
    
                let mut h_counts = 0;
                let mut o_counts = 0;
                let mut hh_trip_counts = 0;
                let mut ho_trip_counts = 0;
                let mut oh_trip_counts = 0;
                let mut oo_trip_counts = 0;
                let mut da_trip_hh = 0.0;
                let mut da_trip_ho = 0.0;
                let mut da_trip_oh = 0.0;
                let mut da_trip_oo = 0.0;

                if t_inf < t_rem && t_rem < trajectory.len() {
                    trajectory = trajectory[t_inf..t_rem as usize].to_vec();

                    for (step_idx, loc) in trajectory.iter().enumerate() {

                        let l = *loc as usize;
                        let t = t_inf + 1 + step_idx;
                        let event_id = event_id_matrix[l][t];

                        match agent.mobility {
                            x if x < 0.05 => {
                                let counter = returner_netmob_per_sim[sim_idx].entry((prev_loc, l)).or_insert(0);
                                *counter += 1;
                            }
                            x if x > 0.305 && x < 0.355 => {
                                let counter = commuter_netmob_per_sim[sim_idx].entry((prev_loc, l)).or_insert(0);
                                *counter += 1;
                            }
                            x if x > 0.95 => {
                                let counter = explorer_netmob_per_sim[sim_idx].entry((prev_loc, l)).or_insert(0);
                                *counter += 1;
                            }
                            _ => {}
                        }

                        if agent.mobility <= 0.1 {
                            a_exp_dist_per_rho[sim_idx][0].push(a_vec[l]);
                        } else if agent.mobility >= 0.9 {
                            a_exp_dist_per_rho[sim_idx][1].push(a_vec[l]);
                        }

                        if l == home {
                            h_counts += 1;
                            cum_a_h += a_vec[l];
                        } else if l != home {
                            o_counts += 1;
                            cum_a_o += a_vec[l];
                        }

                        // Check previous step
                        if prev_loc == home && l == home {
                            hh_trip_counts += 1;
                            da_trip_hh += a_vec[l] - a_vec[prev_loc];
                        } else if prev_loc == home && l != home {
                            ho_trip_counts += 1;
                            da_trip_ho += a_vec[l] - a_vec[prev_loc];
                        } else if prev_loc != home && l == home {
                            oh_trip_counts += 1;
                            da_trip_oh += a_vec[l] - a_vec[prev_loc];
                        } else if prev_loc != home && l != home {
                            oo_trip_counts += 1;
                            da_trip_oo += a_vec[l] - a_vec[prev_loc];
                        }
                        prev_loc = l;

                        // Compute contribution to new infected cases
                        if let Some(event_id_value) = event_id {
                            if event_id_value != 0 {
                                // Find the event in event_ensemble with the given event_id
                                if let Some(event) = event_ensemble.iter().find(|&e| e.id == event_id.unwrap() as u32) {
                                    //if l == home {
                                    //    cum_a_h += a_vec[l];
                                    //} else {
                                    //    cum_a_o += a_vec[l];
                                    //}

                                    if t_rem - t_inf > infectious_period_cutoff {
                                        cum_size += event.size;
                                        cum_pc_foi += event.infected_population as f64 / event.total_population as f64;
                                        cum_foi += event.susceptible_population as f64 * event.infected_population as f64 / event.total_population as f64;
                                        cum_i_pop += event.infected_population;
                                        cum_t_pop += event.total_population;
                                        cum_shared += event.size as f64 / event.infected_population as f64;
                                        nevents_eff += 1;
                                        
                                        // Compute origin-destination of infections
                                        if l == home && out_infection == false {
                                            events_hh_per_rho[sim_idx][rho_idx] += 1;                   
                                        } else if l != home && out_infection == false {
                                            events_ho_per_rho[sim_idx][rho_idx] += 1;
                                        } else if l == home && out_infection {
                                            events_oh_per_rho[sim_idx][rho_idx] += 1;
                                        } else if l != home && out_infection {
                                            events_oo_per_rho[sim_idx][rho_idx] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if t_rem - t_inf > infectious_period_cutoff {
                    let infectious_period = trajectory.len();
                    let h_freq = h_counts as f64 / infectious_period as f64;
                    let o_freq = o_counts as f64 / infectious_period as f64;

                    if h_counts != 0 {
                        let avg_a_h = cum_a_h / h_counts as f64;
                        avg_a_h_dist_per_rho[sim_idx][rho_idx].push(avg_a_h);
                    }
                    if o_counts != 0 {
                        let avg_a_o = cum_a_o / o_counts as f64;
                        avg_a_o_dist_per_rho[sim_idx][rho_idx].push(avg_a_o);
                    }

                    f_inf_tr_h_dist_per_rho[sim_idx][rho_idx].push(h_freq);
                    f_inf_tr_o_dist_per_rho[sim_idx][rho_idx].push(o_freq);

                    let total_trips = hh_trip_counts + ho_trip_counts + oh_trip_counts + oo_trip_counts;
                    if total_trips != 0 {

                        let da_trip_hh = da_trip_hh as f64 / hh_trip_counts as f64;
                        let da_trip_ho = da_trip_ho as f64 / ho_trip_counts as f64;
                        let da_trip_oh = da_trip_oh as f64 / oh_trip_counts as f64;
                        let da_trip_oo = da_trip_oo as f64 / oo_trip_counts as f64;

                        da_trip_hh_dist_per_rho[sim_idx][rho_idx].push(da_trip_hh);
                        da_trip_ho_dist_per_rho[sim_idx][rho_idx].push(da_trip_ho);
                        da_trip_oh_dist_per_rho[sim_idx][rho_idx].push(da_trip_oh);
                        da_trip_oo_dist_per_rho[sim_idx][rho_idx].push(da_trip_oo);

                        let f_hh_trips = hh_trip_counts as f64 / total_trips as f64;
                        let f_ho_trips = ho_trip_counts as f64 / total_trips as f64;
                        let f_oh_trips = oh_trip_counts as f64 / total_trips as f64;
                        let f_oo_trips = oo_trip_counts as f64 / total_trips as f64;

                        f_trip_hh_dist_per_rho[sim_idx][rho_idx].push(f_hh_trips);
                        f_trip_ho_dist_per_rho[sim_idx][rho_idx].push(f_ho_trips);
                        f_trip_oh_dist_per_rho[sim_idx][rho_idx].push(f_oh_trips);
                        f_trip_oo_dist_per_rho[sim_idx][rho_idx].push(f_oo_trips);
                    }
                
                    //let mut avg_a_h = 0.0;
                    //let mut avg_a_o = 0.0;
                    let mut avg_foi = 0.0;
                    let mut avg_pc_foi = 0.0;
                    let mut avg_shared = 0.0;
                    let mut avg_size = 0.0;
                    let mut avg_t_pop = 0.0;
                    if nevents_eff != 0 {
                        //avg_a_h = cum_a_h / nevents_eff as f64;
                        //avg_a_o = cum_a_o / nevents_eff as f64;
                        avg_foi = cum_foi / nevents_eff as f64;
                        avg_pc_foi = cum_pc_foi / nevents_eff as f64;
                        avg_shared = cum_shared / nevents_eff as f64;
                        avg_size = cum_size as f64 / nevents_eff as f64;
                        avg_t_pop = cum_t_pop as f64 / nevents_eff as f64;
                    }

                    avg_foi_per_rho[sim_idx][rho_idx] += avg_foi;
                    avg_pc_foi_per_rho[sim_idx][rho_idx] += avg_pc_foi;
                    avg_shared_per_rho[sim_idx][rho_idx] += avg_shared;
                    avg_size_per_rho[sim_idx][rho_idx] += avg_size;
                    avg_t_pop_per_rho[sim_idx][rho_idx] += avg_t_pop;
                    cum_size_per_rho[sim_idx][rho_idx] += cum_size; 
                    cum_shared_per_rho[sim_idx][rho_idx] += cum_shared; 
                    cum_i_pop_per_rho[sim_idx][rho_idx] += cum_i_pop;
                    cum_t_pop_per_rho[sim_idx][rho_idx] += cum_t_pop; 
                    nevents_eff_per_rho[sim_idx][rho_idx] += nevents_eff;
                }

                }   
        }

        // Compute peak times
        let peak_times_per_loc = find_second_max_indices(&infected_tseries);
        t_peak_dist_per_loc[sim_idx] = peak_times_per_loc;

        // Compute statistics for the distributions within each bin
        for bin_idx in 0..nbins {
            binned_stats_per_sim[sim_idx].t_inv_stats[bin_idx] =
                calculate_stat_pack(&t_inv_dist_per_rho[sim_idx][bin_idx]);
            binned_stats_per_sim[sim_idx].t_inf_stats[bin_idx] =
                calculate_stat_pack(&t_inf_dist_per_rho[sim_idx][bin_idx]);
            binned_stats_per_sim[sim_idx].f_inf_h_stats[bin_idx] =
                calculate_stat_pack(&f_inf_h_dist_per_rho[sim_idx][bin_idx]);
            binned_stats_per_sim[sim_idx].f_inf_o_stats[bin_idx] =
                calculate_stat_pack(&f_inf_o_dist_per_rho[sim_idx][bin_idx]);
            binned_stats_per_sim[sim_idx].a_inf_h_stats[bin_idx] =
                calculate_stat_pack(&a_inf_h_dist_per_rho[sim_idx][bin_idx]);
            binned_stats_per_sim[sim_idx].a_inf_o_stats[bin_idx] =
                calculate_stat_pack(&a_inf_o_dist_per_rho[sim_idx][bin_idx]);
        }

        for loc_idx in 0..nlocs {
            binned_stats_per_sim[sim_idx].inv_rho_stats[loc_idx] = calculate_stat_pack(&invader_rho_per_loc[sim_idx][loc_idx]);
            binned_stats_per_sim[sim_idx].inf_rho_stats[loc_idx] = calculate_stat_pack(&infected_rho_per_loc[sim_idx][loc_idx]);
        }
    }

    // Collect all the output binned results per simulation in a struct
    DigestedOutput {
        agents_per_rho,
        avg_foi_per_rho,
        avg_pc_foi_per_rho,
        avg_shared_per_rho,
        avg_size_per_rho,
        avg_t_pop_per_rho,
        cum_i_pop_per_rho,
        cum_p_exp_per_rho,
        cum_shared_per_rho,
        cum_size_per_rho,
        cum_t_pop_per_rho,
        events_hh_per_rho,
        events_ho_per_rho,
        events_oh_per_rho,
        events_oo_per_rho,
        infected_per_rho,
        infected_h_per_rho,
        infected_o_per_rho,
        invaders_per_rho,
        nevents_eff_per_rho,
        nlocs_invaded,
        total_cases_per_loc,
        binned_stats_per_sim: None,
        a_exp_dist_per_rho: Some(a_exp_dist_per_rho),
        a_inf_h_dist_per_rho: Some(a_inf_h_dist_per_rho),
        a_inf_o_dist_per_rho: Some(a_inf_o_dist_per_rho),
        avg_a_h_dist_per_rho: Some(avg_a_h_dist_per_rho),
        avg_a_o_dist_per_rho: Some(avg_a_o_dist_per_rho),
        da_trip_hh_dist_per_rho: Some(da_trip_hh_dist_per_rho),
        da_trip_ho_dist_per_rho: Some(da_trip_ho_dist_per_rho),
        da_trip_oh_dist_per_rho: Some(da_trip_oh_dist_per_rho),
        da_trip_oo_dist_per_rho: Some(da_trip_oo_dist_per_rho),
        f_inf_h_dist_per_rho: Some(f_inf_h_dist_per_rho),
        f_inf_o_dist_per_rho: Some(f_inf_o_dist_per_rho),
        f_inf_tr_h_dist_per_rho: Some(f_inf_tr_h_dist_per_rho),
        f_inf_tr_o_dist_per_rho: Some(f_inf_tr_o_dist_per_rho),
        f_trip_hh_dist_per_rho: Some(f_trip_hh_dist_per_rho),
        f_trip_ho_dist_per_rho: Some(f_trip_ho_dist_per_rho),
        f_trip_oh_dist_per_rho: Some(f_trip_oh_dist_per_rho),
        f_trip_oo_dist_per_rho: Some(f_trip_oo_dist_per_rho),
        t_inv_dist_per_rho: Some(t_inv_dist_per_rho),
        t_inf_dist_per_rho: Some(t_inf_dist_per_rho),
        infected_rho_per_loc: Some(infected_rho_per_loc),
        infected_rho_h_per_loc: Some(infected_rho_h_per_loc),
        infected_rho_o_per_loc: Some(infected_rho_o_per_loc),
        invader_rho_per_loc: Some(invader_rho_per_loc),
        t_inv_dist_per_loc: Some(t_inv_dist_per_loc),
        t_peak_dist_per_loc: Some(t_peak_dist_per_loc),
        event_output: Some(event_data_per_sim),
        returner_netmob: Some(returner_netmob_per_sim),
        commuter_netmob: Some(commuter_netmob_per_sim),
        explorer_netmob: Some(explorer_netmob_per_sim),
    }
}

// Input-related structs & implementations
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StatPacker {
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub l95: Option<f64>,
    pub u95: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl StatPacker {
    pub fn new(
        mean: Option<f64>,
        std: Option<f64>,
        l95: Option<f64>,
        u95: Option<f64>,
        min: Option<f64>,
        max: Option<f64>,
    ) -> Self {
        Self {
            mean,
            std,
            l95,
            u95,
            min,
            max,
        }
    }
}

pub fn calculate_stat_pack(values: &[f64]) -> StatPacker {
    let mean_tmp = values.iter().sum::<f64>() / values.len() as f64;
    let mean = Some(mean_tmp);
    let variance = Some(values.iter().map(|x| (x - mean_tmp).powi(2)).sum::<f64>() / (values.len() - 1) as f64);
    let std_tmp = variance.unwrap().sqrt();
    let std = Some(std_tmp);
    let z = 1.96; // 95% confidence interval

    let l95 = Some(mean_tmp - z * (std_tmp / (values.len() as f64).sqrt()));
    let u95 = Some(mean_tmp + z * (std_tmp / (values.len() as f64).sqrt()));

    let min = values.iter().min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)).cloned();
    let max = values.iter().max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)).cloned();

    StatPacker::new(mean, std, l95, u95, min, max)
}

#[derive(Debug, Clone, Serialize)]
pub struct BinnedStats {
    pub t_inv_stats: Vec<StatPacker>,
    pub t_inf_stats: Vec<StatPacker>,
    pub f_inf_h_stats: Vec<StatPacker>,
    pub f_inf_o_stats: Vec<StatPacker>,
    pub a_inf_h_stats: Vec<StatPacker>,
    pub a_inf_o_stats: Vec<StatPacker>,
    pub inv_rho_stats: Vec<StatPacker>,
    pub inf_rho_stats: Vec<StatPacker>,
}

impl BinnedStats {
    pub fn new(nbins: usize, nlocs: usize) -> Self {
        BinnedStats {
            t_inv_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nbins],
            t_inf_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nbins],
            f_inf_h_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nbins],
            f_inf_o_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nbins],
            a_inf_h_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nbins],
            a_inf_o_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nbins],
            inv_rho_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nlocs],
            inf_rho_stats: vec![StatPacker::new(Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)); nlocs],
        }
    }
}

// Input-related structs & implementations
#[derive(Debug, Clone, Serialize)]
pub struct DigestedOutput {
    pub agents_per_rho: Vec<Vec<usize>>,
    pub avg_foi_per_rho: Vec<Vec<f64>>,
    pub avg_pc_foi_per_rho: Vec<Vec<f64>>,
    pub avg_shared_per_rho: Vec<Vec<f64>>,
    pub avg_size_per_rho: Vec<Vec<f64>>,
    pub avg_t_pop_per_rho: Vec<Vec<f64>>,
    pub cum_p_exp_per_rho: Vec<Vec<f64>>,
    pub cum_i_pop_per_rho: Vec<Vec<u32>>,
    pub cum_shared_per_rho: Vec<Vec<f64>>,
    pub cum_size_per_rho: Vec<Vec<u32>>,
    pub cum_t_pop_per_rho: Vec<Vec<u32>>,
    pub events_hh_per_rho: Vec<Vec<usize>>,
    pub events_ho_per_rho: Vec<Vec<usize>>,
    pub events_oh_per_rho: Vec<Vec<usize>>,
    pub events_oo_per_rho: Vec<Vec<usize>>,
    pub infected_per_rho: Vec<Vec<usize>>,
    pub infected_h_per_rho: Vec<Vec<usize>>,
    pub infected_o_per_rho: Vec<Vec<usize>>,
    pub invaders_per_rho: Vec<Vec<usize>>,
    pub nevents_eff_per_rho: Vec<Vec<usize>>,
    pub nlocs_invaded: Vec<usize>,
    pub t_inv_dist_per_loc: Option<Vec<Vec<u32>>>,
    pub t_peak_dist_per_loc: Option<Vec<Vec<u32>>>,
    pub total_cases_per_loc: Vec<Vec<u32>>,
    pub binned_stats_per_sim: Option<Vec<BinnedStats>>,
    pub a_exp_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub a_inf_h_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub a_inf_o_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub avg_a_h_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub avg_a_o_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub da_trip_hh_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub da_trip_ho_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub da_trip_oh_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub da_trip_oo_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_inf_h_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_inf_o_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_inf_tr_h_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_inf_tr_o_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_trip_hh_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_trip_ho_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_trip_oh_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub f_trip_oo_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub t_inf_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub t_inv_dist_per_rho: Option<Vec<Vec<Vec<f64>>>>,
    pub infected_rho_per_loc: Option<Vec<Vec<Vec<f64>>>>,
    pub infected_rho_h_per_loc: Option<Vec<Vec<Vec<f64>>>>,
    pub infected_rho_o_per_loc: Option<Vec<Vec<Vec<f64>>>>,
    pub invader_rho_per_loc: Option<Vec<Vec<Vec<f64>>>>,
    pub event_output: Option<Vec<EventOutput>>,
    pub returner_netmob: Option<Vec<HashMap<(usize, usize), u32>>>,
    pub commuter_netmob: Option<Vec<HashMap<(usize, usize), u32>>>,
    pub explorer_netmob: Option<Vec<HashMap<(usize, usize), u32>>>,
}

// Input-related structs & implementations
#[derive(Debug, Clone, Serialize)]
pub struct EventOutput {
    pub attractiveness: Vec<f64>,
    pub inf_pop_avg_rho: Vec<f64>,
    pub infector_rho: Vec<f64>,
    pub location: Vec<u32>,
    pub size: Vec<u32>,
    pub time: Vec<u32>,
    pub tot_pop: Vec<u32>,
}

impl EventOutput {
    pub fn new() -> Self {
        EventOutput {
            attractiveness: Vec::new(),
            inf_pop_avg_rho: Vec::new(),
            infector_rho: Vec::new(),
            location: Vec::new(),
            size: Vec::new(),
            time: Vec::new(),
            tot_pop: Vec::new(),
        }
    }    
}