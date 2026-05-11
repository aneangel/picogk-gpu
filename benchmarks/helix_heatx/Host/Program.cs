// SPDX-FileCopyrightText: Copyright (c) 2026 The picogk-gpu Authors
// SPDX-License-Identifier: Apache-2.0

using Leap71.CoolCube;
using PicoGK;

float voxelSize = 0.5f;

string? envVox = Environment.GetEnvironmentVariable("PICOGK_VOXEL_SIZE");
if (!string.IsNullOrEmpty(envVox) && float.TryParse(envVox, out float vEnv))
{
    voxelSize = vEnv;
}
else if (args.Length >= 1 && float.TryParse(args[0], out float vArg))
{
    voxelSize = vArg;
}

Console.WriteLine($"[picogk-gpu/bench/HelixHeatX] voxel_size={voxelSize}");
PicoGK.Library.Go(voxelSize, HelixHeatX.Task, bEndAppWithTask: true);
