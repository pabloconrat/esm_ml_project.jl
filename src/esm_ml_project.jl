module esm_ml_project

using DynamicalSystems
using OrdinaryDiffEq
using CairoMakie
using StaticArrays
using Printf
using GeoMakie
using SpeedyWeather

export run_simulation

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

end # module
