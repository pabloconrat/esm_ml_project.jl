using esm_ml_project
using Test
using Glob   # for pattern matching files
using Dates
using FilePathsBase: mkpath, rm

@testset "run_simulation basic" begin
    # Run simulation: 5 day spinup, 10 day simulation
    sim = run_simulation(5; sim_days=10)

    # Check output directory for files
    # SpeedyWeather usually creates run_0001/output.nc (or similar)
    run_dirs = sort(Glob.glob("run_*/output.nc"))  # adjust path if needed

    @test !isempty(run_dirs)  # at least one output.nc exists

    # Optionally remove generated files
    for file in run_dirs
        rm(file; force=true)
    end
end
