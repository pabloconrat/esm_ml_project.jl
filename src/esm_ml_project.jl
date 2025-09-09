module esm_ml_project

using DynamicalSystems
using OrdinaryDiffEq
using CairoMakie
using StaticArrays
using Printf
using GeoMakie
using SpeedyWeather
using Statistics
using LinearAlgebra

export run_simulation, prepare_data_batches, prepare_pca_analysis

"""
    run_simulation(spinup_days::Int; sim_days::Int=400, trunc::Int=31, nlayers::Int=8)

Run a Held–Suarez forced dry atmosphere simulation with `SpeedyWeather`.

# Arguments
- `spinup_days`: number of days to run without saving output (spinup).
- `sim_days`: number of days to run with NetCDF output (default = 400).
- `trunc`: spectral truncation T (default = 31).
- `nlayers`: number of vertical layers (default = 8).

# Returns
- `simulation`: the final `Simulation` object from SpeedyWeather.
"""
function run_simulation(spinup_days::Int=100; sim_days::Int=400, trunc::Int=31, nlayers::Int=8)

    # Grid and output setup
    spectral_grid = SpectralGrid(trunc=trunc, nlayers=nlayers)
    output = NetCDFOutput(spectral_grid)

    # Construct dry primitive equations model with Held–Suarez forcing
    model = PrimitiveDryModel(
        spectral_grid,
        output = output,

        # Held–Suarez forcing and drag
        temperature_relaxation = HeldSuarez(spectral_grid),
        boundary_layer_drag = LinearDrag(spectral_grid),

        # switch off other physics
        convection = NoConvection(),
        shortwave_radiation = NoShortwave(),
        longwave_radiation = NoLongwave(),
        vertical_diffusion = NoVerticalDiffusion(),

        # switch off surface fluxes
        surface_wind = NoSurfaceWind(),
        surface_heat_flux = NoSurfaceHeatFlux(),

        # no orography
        orography = EarthOrography(spectral_grid)
    )

    # timestep: 30 minutes at T31 resolution
    model.time_stepping.Δt_at_T31 = Second(1800)

    # Initialize simulation
    simulation = SpeedyWeather.initialize!(model)

    # Run spinup (no output)
    run!(simulation, period=Day(spinup_days))

    # Run main simulation with output
    run!(simulation, period=Day(sim_days), output=true)

    return simulation
end

"""
    prepare_data_batches(t::AbstractVector, X::AbstractMatrix; train_frac=0.8, batchsize=10)

Splits data into train and validation sets and returns batches & separation index:

# Returns
- `train_batches::Vector{Tuple{Vector, Matrix}}`
- `valid_batches::Vector{Tuple{Vector, Matrix}}`
- `split_idx::Int64``
"""
function prepare_data_batches(t::AbstractVector, X::AbstractMatrix; train_frac=0.8, batchsize=10)
    N = size(X, 1)  # number of time points
    split_idx = Int(floor(N * train_frac))

    t_train, t_valid = t[1:split_idx], t[split_idx+1:end]
    X_train, X_valid = X[1:split_idx, :], X[split_idx+1:end, :]

    function make_batches(t::AbstractVector, X::AbstractMatrix, batchsize::Int)
        N = size(X, 1)  # number of columns = time points
        batches = []

    for start in 1:2:N
        stop = start + batchsize - 1
        if stop > N
            break  # exit loop if the batch would go past N
        end
        t_batch = t[start:stop]
        X_batch = X[start:stop, :]
        push!(batches, (t_batch, X_batch))
    end

        return batches
    end

    train_batches = make_batches(t_train, X_train, batchsize)
    valid_batches = make_batches(t_valid, X_valid, batchsize)

    return train_batches, valid_batches, split_idx
end

"""
    prepare_pca_analysis(u_jet::Array{<:Real,3}, K::Int)

Perform PCA/EOF analysis on the input field `u_jet` (nlon, nlat, nt).

# Arguments
- `u_jet` : 3D array of the field with dimensions `(nlon, nlat, nt)`.
- `K`     : Number of leading EOFs/PCs to retain.

# Returns
- `A`         : (K × time) array of retained principal component time series.
- `PCs`       : (time × space) array of full PCs.
- `EOFs`      : (space × K) matrix of leading EOF patterns.
- `X_anom`    : (time × space) anomaly matrix.
- `X_timemean`: (1 × space) temporal mean field at each gridpoint.
- `frac_var`  : fraction of variance explained by each EOF.
"""
function prepare_pca_analysis(u_jet::AbstractArray{<:Real,3}, K::Int)
    nlon, nlat, nt = size(u_jet)
    
    # reshape to (space, time)
    X = reshape(u_jet, nlon*nlat, nt)
    # transpose: (time, space)
    X = permutedims(X)
    
    # mean across time at each gridpoint
    X_timemean = mean(X, dims=1)
    
    # anomalies
    X_anom = X .- X_timemean
    
    # covariance matrix in space
    C = X_anom' * X_anom / (size(X_anom, 1) - 1)
    
    # eigendecomposition
    eig_vals, eig_vecs = eigen(C, sortby=-)
    
    # principal components
    PCs = X_anom * eig_vecs
    
    # fraction of variance explained
    frac_var = eig_vals ./ sum(eig_vals)
    
    # truncate to K
    A = PCs[:, 1:K]'            # (K × time)
    EOFs = eig_vecs[:, 1:K]     # (space × K)
    
    return A, PCs, EOFs, X_anom, X_timemean, frac_var
end


end # module
