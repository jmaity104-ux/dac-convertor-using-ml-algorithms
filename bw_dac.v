`timescale 1ns / 1ps

module bw_dac #(parameter N = 12) (
    input  wire [N-1:0] din,
    output [63:0] vout_bits
);

    parameter real VREF = 1.2;
    real weights [N-1:0];
    real vout;
    integer i;
    integer seed_val;
    integer rand_val;
    real rand_factor;

    initial begin
        seed_val = 1234;
        for (i = 0; i < N; i = i + 1) begin
            rand_val = $urandom(seed_val);  // Icarus-safe call
            seed_val = seed_val + 1;        // change seed each time
            rand_factor = 1.0 + ((rand_val % 1000 - 500) / 100000.0);
            weights[i] = (1.0 / (2.0 ** (N - i))) * rand_factor;
        end
    end

    always @(*) begin
        vout = 0.0;
        for (i = 0; i < N; i = i + 1)
            if (din[i])
                vout = vout + weights[i] * VREF;
    end

    assign vout_bits = $realtobits(vout);

endmodule
