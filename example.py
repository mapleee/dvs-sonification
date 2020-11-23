import numpy as np
import fluidsynth
import pretty_midi
from midi2audio import FluidSynth

# sf2_path = 'data/sf2/gnusmas_gm_soundfont_2.00.sf2'
sf2_path = 'data/sf2/GeneralUser GS 1.442 MuseScore/GeneralUser GS MuseScore v1.442.sf2'

midi_data = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=0)

# note = pretty_midi.Note(velocity=100, pitch=48, start=0, end=3)
# piano.notes.append(note)
# for i in range(3):
#     event = pretty_midi.ControlChange(number=10, value=i*63, time=i)
#     piano.control_changes.append(event)

for i in range(3):
    note = pretty_midi.Note(velocity=100, pitch=48+i*12, start=i, end=1+i)
    event = pretty_midi.ControlChange(number=10, value=i*63, time=i)
    piano.notes.append(note)
    piano.control_changes.append(event)

midi_data.instruments.append(piano)
midi_data.write('example_1.mid')
sf2 = FluidSynth(sf2_path)
sf2.midi_to_audio('example_1.mid', 'example_1.wav')


