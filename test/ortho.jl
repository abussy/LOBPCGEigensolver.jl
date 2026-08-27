@testset "ortho! produces orthonormal columns" begin
    for T in (Float64, ComplexF64)
        X = rand(T, 30, 6)
        Y = ortho!(copy(X))[1]
        @test norm(Y' * Y - I) < 1e-10
    end
end

@testset "ortho! against a subspace (B = I)" begin
    Y = ortho!(rand(40, 5))[1]        # already orthonormal
    X = rand(40, 4)
    ortho!(X, Y, Y)                   # make X orthonormal and ⟂ Y
    @test norm(X' * X - I) < 1e-9
    @test norm(Y' * X) < 1e-9
end

@testset "Duersch search directions" begin
    for T in (Float64, ComplexF64, Float32)
        Z = Matrix(qr(randn(T, 30, 30)).Q)
        cX = @view Z[:, 1:10]
        cXperp = @view Z[:, 11:end]
        active = 3:10
        compact = LOBPCGEigensolver.get_new_P_duersch(
            cX, cXperp, active; tol=2eps(real(T)),
            timer=LOBPCGEigensolver.disabled_timer
        )
        selector = LOBPCGEigensolver.get_new_P(
            cX, active; tol=2eps(real(T)), timer=LOBPCGEigensolver.disabled_timer
        )
        tolerance = 100eps(real(T))
        @test norm(cX' * compact) < tolerance
        @test norm(compact' * compact - I) < tolerance
        @test norm(compact - selector) < tolerance
    end

    cX = zeros(8, 2)
    cX[1:2, :] .= [1 1; 1 -1] / sqrt(2)
    cXperp = zeros(8, 6)
    cXperp[3:8, :] .= Matrix(I, 6, 6)
    cP = LOBPCGEigensolver.get_new_P_duersch(
        cX, cXperp, 1:2; tol=2eps(), timer=LOBPCGEigensolver.disabled_timer
    )
    @test norm(cX' * cP) < 1e-14
    @test norm(cP' * cP - I) < 1e-14
end
