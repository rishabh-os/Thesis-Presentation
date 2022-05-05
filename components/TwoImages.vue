<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps({
  img1: String,
  img2: String,
  desc1: String,
  desc2: String,
})

const img1 = ref(props.img1)
const img2 = ref(props.img2)
const desc1 = ref(props.desc1)
const desc2 = ref(props.desc2)
</script>

<template>
  <br><br>
  <div class="grid grid-cols-2 text-center divide-x-2 divide-green-500">
    <div>
      <div class="flex items-center justify-center mb-2">
        <img :src="img1">
      </div>
      {{ desc1 }}
    </div>
    <div>
      <div class="flex items-center justify-center mb-2">
        <img :src="img2">
      </div>
      {{ desc2 }}
    </div>
  </div>
</template>
