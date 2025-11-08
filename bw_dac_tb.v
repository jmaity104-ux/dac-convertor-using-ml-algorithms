`timescale 1ns / 1ps

module bw_dac_tb;
    parameter N = 12;
    parameter real VREF = 1.2;
    real lsb;
    reg [N-1:0] din;
    wire real vout;
    real vout_prev, dnl, inl, ideal_vout;
    integer code, f;
    real delay_ns;
    wire [63:0] vout_bits;

    bw_dac #(N) dut (
        .din(din),
        .vout_bits(vout_bits)
    );

    assign vout = $bitstoreal(vout_bits);

    initial begin
        lsb = VREF / ((1 << N) - 1);
        vout_prev = 0.0;
        delay_ns = 100.0;

        f = $fopen("dac_output_varied_bits.csv", "w");
        $fwrite(f, "code,b11,b10,b9,b8,b7,b6,b5,b4,b3,b2,b1,b0,vout,DNL,INL,delay_ns\n");

        for (code = 0; code < (1 << N); code = code + 1) begin
            din = code;
            #125;

            ideal_vout = code * lsb;

            if (code == 0) begin
                dnl = 0.0;
                inl = 0.0;
            end else begin
                dnl = ((vout - vout_prev) / lsb) - 1.0;
                inl = (vout - ideal_vout) / lsb;
            end
            vout_prev = vout;

            $fwrite(f,
                "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0.6f,%0.6f,%0.6f,%0.1f\n",
                code,
                din[11], din[10], din[9], din[8],
                din[7], din[6], din[5], din[4],
                din[3], din[2], din[1], din[0],
                vout, dnl, inl, delay_ns
            );
        end

        $fclose(f);
        $display("✅ Dataset generated: dac_output_varied_bits.csv");
        $finish;
    end
endmodule
