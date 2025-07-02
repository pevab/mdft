using DrWatson
@quickactivate "mdft"

using Distributions

abstract type  AttentionProcess
end

struct BernoulliAttentionProcess <: AttentionProcess
    probabilities :: Vector{Float64}
end

function next(process :: BernoulliAttentionProcess) :: Vector{Float64}
    probabilities = process.probabilities
    n_attributes = length(probabilities)
    result = zeros(n_attributes)
    for a in 1:n_attributes
        p = probabilities[a]
        result[a] = rand(Bernoulli(p))
    end
    result
end


struct MDFT 
    n_alternatives :: Int64
    n_attributes :: Int64
    personal_evaluation :: Matrix{Float64}
    attention_process :: AttentionProcess
    contrast_matrix :: Matrix{Float64}
    feedback_matrix :: Matrix{Float64}
end