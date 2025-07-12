
using DrWatson
@quickactivate "mdft"

using LinearAlgebra
using Plots
using NNlib
using Distributions

#= -- MLBA
We focus on 2 attributes (E, Q).
=#

#=
MLBA STRUCTURE
=#

struct MLBA
    n_alternatives :: Int
    w_e :: Float64
    phi :: Float64
    beta :: Float64
    gamma :: Float64
    objective_evaluation :: Matrix{Float64} # n_alternatives x 2
    A :: Float64
    s :: Float64
end

function attention_weight(mlba :: MLBA)
    return [mlba.w_e, 1 - mlba.w_e]
end

function evaluation_matrix(mlba :: MLBA)
    v_e = mlba.objective_evaluation[:, 1]
    v_q = mlba.objective_evaluation[:, 2]
    m_e = v_e / maximum(v_e)
    m_q = v_q / maximum(v_q)
    [m_e m_q]
end

function psy_dist2(mlba :: MLBA, a :: Int, b :: Int )
    m = evaluation_matrix(mlba)
    de = m[a, 1] - m[b, 1]
    dq = m[a, 2] - m[b, 2]
    di = (de - dq)/sqrt(2)
    dd = (de + dq)/sqrt(2)
    di^2 + mlba.beta * dd^2
end

function contrast_matrix(mlba :: MLBA)
    n = mlba.n_alternatives
    phi = mlba.phi
    alpha = zeros((n, n))
    for i in 1:n
        for j in 1:n
            if i != j
                alpha[i,j] = 1 - exp(-phi * sqrt(psy_dist2(mlba, i, j)))
            end
        end
    end
    Matrix(I, (n,n)) - 0.5 * alpha
end

function valence_vector(mlba :: MLBA)
    C = contrast_matrix(mlba)
    M = evaluation_matrix(mlba)
    W = attention_weight(mlba)
    C * M * W
end

function mean_drift(mlba :: MLBA)
    n = mlba.n_alternatives
    gamma = mlba.gamma
    v = valence_vector(mlba)
    (10 * ones(n)) ./ (ones(n) + exp.(- gamma * v))
end

function sample(mlba :: MLBA, chi :: Float64)
    n = mlba.n_alternatives
    A = mlba.A
    s = mlba.s
    starting_point = rand(Uniform(0, A), n)
    mean_drift_rate = mean_drift(mlba)
    drift_rate = rand(MultivariateNormal(mean_drift_rate, s * Matrix(I, n,n)))
    t = (chi * ones(n) - starting_point) ./ drift_rate
    _, w = findmin(t)
    r = zeros(n)
    r[w] = 1
    r
end

#=
MEASURE
=#

struct Measure
    model :: MLBA
    n_seeds :: Int
    min_threshold :: Float64
    max_threshold :: Float64
    n_steps :: Int
end

struct MeasureResult
    mean :: Float64
    std :: Float64
end

function run_measure(measure :: Measure) :: Matrix{MeasureResult}
    model = measure.model
    min_threshold = measure.min_threshold
    max_threshold = measure.max_threshold
    n = model.n_alternatives
    n_steps = measure.n_steps
    n_seeds = measure.n_seeds
    results = Matrix{MeasureResult}(undef, n, n_steps)
    for step in 1:n_steps
        chi = min_threshold + step * (max_threshold - min_threshold) / n_steps
        s = reduce(hcat, [sample(model, chi) for _ in 1:n_seeds])
        s_mean = mean(s, dims=2)
        s_std = std(s, dims=2)
        results[:, step] = [MeasureResult(s_mean[a], s_std[a]) for a in 1:n]
    end
    results
end

#=
EXAMPLES
=#

#=
SIMILARITY EFFECT
=#

function similarity_effect()
    min_threshold = 0.0
    max_threshold = 10.0
    n_steps = 100
    n_seeds = 1000
    w_e = 0.5
    phi = 100.0
    beta = 10.0
    gamma = 0.1
    obj_m = [ # objective evaluation matrix
                # E     Q
                1.00  3.00; # A
                0.50  1.50; # S
                3.00  1.00; # B
    ]
    A = 0.001
    s = 0.001
    measure_ab = Measure(
        MLBA(
            2,
            w_e,
            phi,
            beta,
            gamma,
            obj_m[[1,3], :],
            A,
            s
        ),
        n_seeds,
        min_threshold,
        max_threshold,
        n_steps,
    )
    measure_asb = Measure(
        MLBA(
            3,
            w_e,
            phi,
            beta,
            gamma,
            obj_m,
            A,
            s
        ),
        n_seeds,
        min_threshold,
        max_threshold,
        n_steps,
    )

    # Alternatives scatter plot
    scatter_asb = plot()
    scatter!(obj_m[:,1]', obj_m[:, 2]', title="Alternatives", 
        legend=true, 
        label=["A" "S" "B"], 
        color=[:blue :red :green], 
        xlim=(0,3.5), ylim=(0, 3.5),
        alpha=0.5)

    # AB
    results_ab = run_measure(measure_ab)
    mean_ab = map(r -> r.mean, results_ab)
    std_ab = map(r -> r.std, results_ab)
    plt_ab = plot(title="Control")
    colors_ab = [:blue :green]
    labels_ab = ['A' 'B']
    for i in 1:2
        plot!(range(min_threshold, max_threshold, n_steps), mean_ab[i, :], ribbon=0.67 * std_ab[i, :], linecolor=colors_ab[i], label=labels_ab[i], ylim=(0,1))
    end

    # ASB
    results_asb = run_measure(measure_asb)
    mean_asb = map(r -> r.mean, results_asb)
    std_asb = map(r -> r.std, results_asb)
    plt_asb = plot(title="Similarity")
    colors_asb = [:blue :red :green]
    labels_asb = ['A' 'S' 'B']
    for i in 1:3
        plot!(range(min_threshold, max_threshold, n_steps), mean_asb[i, :], ribbon=0.67 * std_asb[i, :], linecolor=colors_asb[i], label=labels_asb[i], ylim=(0,1))
    end

    # plot
    plot(scatter_asb, plt_ab, plt_asb, layout=(1,3))
end