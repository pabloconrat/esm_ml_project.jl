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

using Test

@testset "prepare_data_batches with incomplete last batch" begin
    # Dummy data
    N = 13  # total time points, chosen so last batch would exceed N
    K = 3   # features
    t = collect(1:N)
    X = reshape(1:(N*K), N, K)  
    
    batchsize = 4
    train_frac = 0.8

    train_batches, valid_batches, split_idx = prepare_data_batches(t, X; train_frac=train_frac, batchsize=batchsize)

    # Check split index
    @test split_idx == Int(floor(N * train_frac))

    # Ensure no batch exceeds matrix bounds
    for (t_batch, X_batch) in train_batches
        @test length(t_batch) == size(X_batch, 1) <= batchsize
        @test size(X_batch, 2) == K
    end

    for (t_batch, X_batch) in valid_batches
        @test length(t_batch) == size(X_batch, 1) <= batchsize
        @test size(X_batch, 2) == K
    end

    # Check first batch values
    @test train_batches[1][1] == t[1:batchsize]
    @test train_batches[1][2] == X[1:batchsize, :]

    # Ensure last batch does not exceed bounds
    @test last(train_batches)[1][end] <= split_idx
end
